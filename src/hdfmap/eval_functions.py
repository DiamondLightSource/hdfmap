"""
hdf eval functions
"""

import os
import ast
import builtins
import re
import typing
import numpy as np
import h5py

# parameters
GLOBALS = {'np': np}
GLOBALS_NAMELIST = dir(builtins) + list(GLOBALS.keys())


def shortstr(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    #return re.sub(r'(\d\d\d)\d{4,}', r'\1', string)
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re.sub(r'\d+\.\d{5,}', subfun, string)


def check_naughty_eval(eval_str: str) -> None:
    """
    Check str for naughty eval arguments such as sys, os or import
    This is not foolproof.
    :param eval_str: str
    :return: pass or raise error
    """
    bad_names = ['import', 'os.', 'sys.']
    for bad in bad_names:
        if bad in eval_str:
            raise Exception('This operation is not allowed as it contains: "%s"' % bad)


def find_varnames(expression: str) -> list[str]:
    """Returns list of variable names in expression, ommiting builtins and globals"""
    # varnames = re.findall(r'[a-zA-Z]\w*', expression)
    return [node.id for node in ast.walk(ast.parse(expression, mode='eval'))
            if type(node) is ast.Name and node.id not in GLOBALS_NAMELIST]


def extra_hdf_data(hdf_file: h5py.File) -> dict:
    """Extract filename, filepath and other additional data fom hdf file"""
    return {
        'filepath': hdf_file.filename if hasattr(hdf_file, 'filename') else 'unknown',
        'filename': os.path.basename(hdf_file.filename) if hasattr(hdf_file, 'filename') else 'unknown',
    }


def generate_namespace(hdf_file: h5py.File, hdf_namespace: dict[str, str], varnames: list[str] | None = None,
                       default: typing.Any = np.array('--')) -> dict[str, typing.Any]:
    """
    Generate namespace dict - create a dictionary linking the name of a dataset to the dataset value

    Adds additional values if not in name_address dict:
        filename: str, name of hdf_file
        filepath: str, full path of hdf_file
        _*name*: str hdf address of *name*

    :param hdf_file: h5py.File object
    :param hdf_namespace: dict[varname]='/hdf/dataset/address'
    :param varnames: list of str or None, if None, use generate all items in name_address
    :param default: any, if varname not in name_address - return default instead
    :return: dict {'name': value, '_name': '/hdf/address'}
    """
    if varnames is None:
        varnames = list(hdf_namespace.keys())
    namespace = {name: hdf_file[hdf_namespace[name]][()] for name in varnames if name in hdf_namespace}
    defaults = {name: default for name in varnames if name not in hdf_namespace}
    addresses = {'_' + name: hdf_namespace[name] for name in varnames if name in hdf_namespace}
    # add extra params
    extras = extra_hdf_data(hdf_file)
    return {**defaults, **extras, **addresses, **namespace}


def eval_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str], debug: bool = False) -> typing.Any:
    """
    Evaluate an expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression to be evaluated
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/address'}
    :param debug: bool, if True, returns additional info
    :return: eval(expression)
    """
    if expression in hdf_file:
        return hdf_file[expression][()]
    check_naughty_eval(expression)
    varnames = [name for name in hdf_namespace if name in expression]  # find varnames matching map
    varnames += find_varnames(expression)  # finds other varnames (not builtins)
    namespace = generate_namespace(hdf_file, hdf_namespace, varnames)
    if debug:
        print(f"Expression: {expression}\nvarnames: {varnames}\nnamespace: {namespace}\n")
    return eval(expression, GLOBALS, namespace)


def format_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str], debug: bool = False) -> str:
    """
    Evaluate a formatted string expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression using {name} format specifiers
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/address'}
    :param debug: bool, if True, returns additional info
    :return: eval_hdf(f"expression")
    """
    expression = 'f"""' + expression + '"""'  # convert to fstr
    return eval_hdf(hdf_file, expression, hdf_namespace, debug)


