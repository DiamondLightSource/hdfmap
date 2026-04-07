"""
Specific tests for eval_functions.py
"""

import os
import hdfmap
from hdfmap.eval_functions import replace_expression_vars, prepare_expression_load_data


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_HKL = DATA_FOLDER + "/1049598.nxs"  # hkl scan, pilatus


def test_replace_expression_vars():
    expression = "a + data_a + max_value"
    mapping = {"a": "x", "data": 'y', "max_value": "max(x)"}
    new_expr = replace_expression_vars(expression, mapping)
    assert new_expr == "x + data_a + max(x)"


def test_prepare_expression_load_data():
    expression = 'idgap@units'
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        new_expr = prepare_expression_load_data(hdf, expression, {}, {}, {})
    assert new_expr == expression



