"""
Specific tests for eval_functions.py
"""

import os
import hdfmap
from hdfmap.eval_functions import (replace_expression_vars, prepare_expression_load_data,
                                   eval_hdf, find_header)


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_HKL = DATA_FOLDER + "/1040323.nxs"  # hkl scan, pilatus


def test_replace_expression_vars():
    expression = "a + data_a + max_value"
    mapping = {"a": "x", "data": 'y', "max_value": "max(x)"}
    new_expr = replace_expression_vars(expression, mapping)
    assert new_expr == "x + data_a + max(x)"

    expression = "(cmd|command|scan_command?(''))\nstr((cmd|command|scan_command?('no_cmd')))\ncmd"
    mapping = {'cmd': 'path'}
    new_expr = replace_expression_vars(expression, mapping)
    assert new_expr == "(cmd|command|scan_command?(''))\nstr((cmd|command|scan_command?('no_cmd')))\npath"

    expression = "max(_t)"
    mapping = {'_t': '(count_time|counttime|t?(1.0))'}
    new_expr = replace_expression_vars(expression, mapping)
    assert new_expr == "max((count_time|counttime|t?(1.0)))"


def test_prepare_expression_load_data():
    expression = 'idgap@units, x*y'
    repl = {
        'x': 'a',
    }
    hdf_map = {'idgap': '/entry/instrument/insertion_device/gap'}
    data = {}
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        new_expr = prepare_expression_load_data(hdf, expression, hdf_map, data, repl)
    assert new_expr == "attr__idgap_units, a*y"
    assert data['attr__idgap_units'] == 'mm'


def test_find_header():
    expression = """
    ***
    def myfun(a, b):
        return arange(a, b)
    ***
    myfun(5, 6).tolist()
    """
    header, expr = find_header(expression)
    assert expr == 'myfun(5, 6).tolist()'


def test_header_expression():
    expression = """
    ***
    def myfun(a, b):
        return arange(a, b)
    ***
    myfun(5, 10).tolist()
    """
    hdf_map = {'idgap': '/entry/instrument/insertion_device/gap'}
    repl = {
        'a': 'x',
    }
    data = {}
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        result = eval_hdf(hdf, expression, hdf_map, data, repl)
    assert result == [5, 6, 7, 8, 9]

