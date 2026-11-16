"""
hdf eval functions
"""

import sys
import os
import ast
import asteval
import re
import typing
import numpy as np
import h5py

from .base_functions import generate_identifier, dataset2data, dataset2str, dataset_attribute
from .objects import Dataset
from .logging import create_logger

# parameters
GLOBALS_NAMELIST = asteval.make_symbol_table(use_numpy=True).keys()
PROTECTED = ['filepath', 'filename']
DEFAULT: typing.Any = np.array('--')  # default return in eval
logger = create_logger(__name__)
# regex patterns
re_dataset_attributes = re.compile(r'([a-zA-Z_]\w*)@([a-zA-Z_]\w*)')  # finds 'name@attribute' in expressions
re_dataset_default = re.compile(r'(\w+)\?\((.+?)\)')  # finds 'name?('noname'), return (name, 'noname')
# fromisoformat requires python 3.11+

if sys.version_info < (3, 11, 0):
    logger.warning("Nexus timestamps are not convertable by datetime.fromisoformat in python version <3.11")


def find_identifiers(expression: str) -> list[str]:
    """Returns list of variable names in expression"""
    return [name for name in asteval.get_ast_names(ast.parse(expression))]


def find_or_expressions(expr: str) -> set[str]:
    """
    Returns instances of OR-expressions within the expression,

    An OR-expression is given as '(a|b|c)'

    Brackets are counted correctly so only the OR-expression will be returned.

    :param expr: string expression
    :return: set of unique string expressions
    """
    stack = []
    alternates = []

    for i, ch in enumerate(expr):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                continue
            start = stack.pop()
            sub_str = expr[start:i + 1]
            if '|' in sub_str:
                alternates.append(sub_str)
                stack.clear()
    return set(alternates)


def replace_or_expressions(expr: str, path_map: dict[str, str], path_check: h5py.File | list | dict,
                           data: dict[str, typing.Any]) -> str:
    """
    Replaces instances of (a|b|c)-OR-expressions within the string.
    The first item of the OR-expression that is found in data or if the
    associated path is found in path_check will replace the expression.

        path_map = {'x': '/entry/data/x', 'y': '/entry/data/y'}
        path_check = h5py.File('mydata.h5')  # file contains x path but not y path
        expression = '(y|x)'
        new = replace_or_expressions(expression, path_map, path_check, {})
        # new == 'x'

    :param expr: string expression with components to replace
    :param path_map: dict mapping from variable name to path as found in path_check
    :param path_check: Hdf file to check path in, or could be a list or dict of paths.
    :param data: dict mapping of variable name to data
    :return: expression with replaced components
    """
    while or_expressions := find_or_expressions(expr):
        for or_exp in or_expressions:
            names = or_exp.strip('()').split('|')
            name = next(
                (n for n in names if n in data),
                next((n for n in names if path_map.get(n, '') in path_check), names[-1])
            )
            expr = expr.replace(or_exp, name)
    return expr


def replace_expression_vars(expr: str, mapping: dict[str, str]) -> str:
    """
    Replace variable names in an expression

        'x + y' = replace_expression_vars('a + b', {'a': 'x', 'b': 'y'})

    :param expr: expression to replace variable names
    :param mapping: mapping from variable names to replacements
    :return: replaced expression
    """
    # Find OR expressions (a|b), don't replace anything in these
    sub_expr = find_or_expressions(expr)

    for n, sub_str in enumerate(sub_expr):
        expr = expr.replace(sub_str, f"__ALT{n}__")

    # Replace names in mapping
    for name, repl in mapping.items():
        pattern = r'\b' + re.escape(name) + r'\b'
        expr = re.sub(pattern, repl, expr)

    # recover original expression parts
    for n, sub_str in enumerate(sub_expr):
        expr = expr.replace(f"__ALT{n}__", sub_str)
    return expr


def replace_defaults(expr: str, path_map: dict[str, str]) -> str:
    """
    Replace default specifiers in an expression
    A default specifier is given as 'a?(b)', where a will be
    returned if 'a' is in path_map, otherwise 'b' is returned.

        path_map = {'x': '/entry/data/x', 'y': '/entry/data/y'}
        expression = 'x + y?(10) + z?(20)'
        result = replace_defaults(expression, path_map)
        # result == 'x + y + 20'

    :param expr: string expression with components to replace
    :param path_map: dict mapping from variable name to path
    :return: expression with replaced components
    """
    for match in re_dataset_default.finditer(expr):
        name, name_default = match.groups()
        if name.isnumeric() or name in path_map:
            expr = expr.replace(match.group(), name)
        else:
            expr = expr.replace(match.group(), name_default)
    return expr


def replace_attributes(expr: str, path_map: dict[str, str],
                       hdf_file: h5py.File | dict[str, Dataset]) -> tuple[str, dict]:
    """
    Find sub-strings of the form 'name@attribute', and replace this with
    the attribute associated with the dataset specified by name.

        path_map = {'x': '/entry/data/x', 'y': '/entry/data/y'}
        hdf_file = h5py.File('mydata.h5'), where dataset('/entry/data/x') as attribute 'units'='mm'
        expression = 'x x@units'
        result = replace_attributes(expression, path_map, hdf_file)
        # result == 'x mm'

    :param expr: string expression with components to replace
    :param path_map: dict mapping from variable name to path
    :param hdf_file: h5py.File object
    :return: expression with replaced components
    """
    attributes = {
        f"attr__{name}_{attr}": dataset_attribute(hdf_file[path], attr)
        for name, attr in re_dataset_attributes.findall(expr)  # name@attr
        if (path := path_map.get(name, '')) in hdf_file
    }
    # replace name@attribute in expression
    expr = re_dataset_attributes.sub(r'attr__\g<1>_\g<2>', expr)
    return expr, attributes


def extra_hdf_data(hdf_file: h5py.File) -> dict:
    """Extract filename, filepath and other additional data fom hdf file"""
    filepath = getattr(hdf_file, 'filename', 'unknown')
    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
    }


def generate_namespace(hdf_file: h5py.File, hdf_namespace: dict[str, str],
                       identifiers: list[str] | None = None, default: typing.Any = DEFAULT,
                       data_namespace: dict[str, typing.Any] | None = None) -> dict[str, typing.Any]:
    """
    Generate namespace dict - create a dictionary linking the name of a dataset to the dataset value

    Adds additional values if not in name_path dict:
        filename: str, name of hdf_file
        filepath: str, full path of hdf_file
        _*name*: str hdf path of *name*
        __*name*: str internal name of *name* (e.g. for 'axes')
        s_*name*: string representation of dataset
        d_*name*: return dataset object **warning**: may result in file not closing on completion
        *name*@attr: returns attribute of dataset *name*

    :param hdf_file: h5py.File object
    :param hdf_namespace: locations of data in hdf file, dict[identifier]='/hdf/dataset/path'
    :param identifiers: list of names to load from hdf_file, if None, use generate all items in name_path
    :param default: any, if varname not in name_path - return default instead
    :param data_namespace: dict of {'variable name': value} ** note: values will be added to this dict
    :return: dict {'name': value, '_name': '/hdf/path'}
    """
    if identifiers is None:
        identifiers = list(hdf_namespace.keys())
    if data_namespace is None:
        data_namespace = {}

    def select_ids(startswith=''):
        return (
            (symbol, name) for symbol in identifiers
            if symbol.startswith(startswith) and hdf_namespace.get(name := symbol[len(startswith):], '') in hdf_file
        )

    extra_data = extra_hdf_data(hdf_file)
    namespace = {symbol: dataset2data(hdf_file[hdf_namespace[name]]) for symbol, name in select_ids()}
    strings = {symbol: dataset2str(hdf_file[hdf_namespace[name]], units=True) for symbol, name in select_ids('s_')}
    datasets = {symbol: hdf_file[hdf_namespace[name]] for symbol, name in select_ids('d_')}
    hdf_paths = {symbol: hdf_namespace[name] for symbol, name in select_ids('_')}
    hdf_names = {symbol: generate_identifier(hdf_namespace[name]) for symbol, name in select_ids('__')}
    # generate defaults for non-builtin names that are not in the file
    defaults = {
        name: default
        for name in identifiers
        if name not in data_namespace and name not in GLOBALS_NAMELIST
           and hdf_namespace.get(name, '') not in hdf_file
    }
    # combine with precedence
    data = {**defaults, **hdf_paths, **hdf_names, **datasets, **strings, **namespace, **extra_data}
    # add or overwrite data in data_namespace
    data_namespace.update(data)
    return data


def prepare_expression(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
                       data_namespace: dict[str, typing.Any] | None, replace_names: dict[str, str]) -> str:
    """
    Prepare an expression for evaluation using the namespace of the hdf file
    Returns the modified expression replacing attribute names and alternates with
    valid identifiers. Also updates the data_namespace dict with attribute data.

    The following patterns are allowed in the expression:
     - 'filename': str, name of hdf_file
     - 'filepath': str, full path of hdf_file
     - '_*name*': str hdf path of *name*
     - '__*name*': str internal name of *name* (e.g. for 'axes')
     - 's_*name*': string representation of dataset (includes units if available)
     - 'd_*name*': return dataset object. **warning**: may result in file not closing on completion
     - '*name*@attr': returns attribute of dataset *name*
     - '*name*?(default)': returns default if *name* doesn't exist
     - '(name1|name2|name3)': returns the first available of the names
     - '(name1|name2?(default))': returns the first available name or default

    Additional variables can be added to the evaluation local namespace using data_namespace.

    Shorthand variables for expressions can be assigned using replace_names = {'new_name': 'favourite*expression'}

    :param hdf_file: h5py.File object
    :param expression: str expression to be evaluated
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param data_namespace: dict of {'variable name': value} ** note: values will be added to this dict
    :param replace_names: dict of {'variable_name': expression}
    :return: str expression
    """
    if data_namespace is None:
        data_namespace = {}  # note the link to the parent namespace must be preserved
    # replace names with expressions
    expression = replace_expression_vars(expression, replace_names)
    # replace name@attribute in expression and add values to data_namespace
    expression, attributes = replace_attributes(expression, hdf_namespace, hdf_file)
    data_namespace.update(attributes)  # adds data in the parent function
    # find values with defaults '..?(..)'
    expression = replace_defaults(expression, hdf_namespace)
    # find alternate names '(opt1|opt2|opt3)'
    expression = replace_or_expressions(expression, hdf_namespace, hdf_file, attributes)
    # find values with defaults '..?(..)'
    expression = replace_defaults(expression, hdf_namespace)
    return expression


def prepare_expression_load_data(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
                                 data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
                                 default: typing.Any = DEFAULT, use_stored_data: bool = False):
    """
    Prepare an expression for evaluation using the namespace of the hdf file
    Returns the modified expression replacing attribute names and alternates with
    valid identifiers. Also updates the data_namespace dict with attribute data and data from the hdf file.

    The following patterns are allowed in the expression:
     - 'filename': str, name of hdf_file
     - 'filepath': str, full path of hdf_file
     - '_*name*': str hdf path of *name*
     - '__*name*': str internal name of *name* (e.g. for 'axes')
     - 's_*name*': string representation of dataset (includes units if available)
     - 'd_*name*': return dataset object. **warning**: may result in file not closing on completion
     - '*name*@attr': returns attribute of dataset *name*
     - '*name*?(default)': returns default if *name* doesn't exist
     - '(name1|name2|name3)': returns the first available of the names
     - '(name1|name2?(default))': returns the first available name or default

    Additional variables can be added to the evaluation local namespace using data_namespace.

    Shorthand variables for expressions can be assigned using replace_names = {'new_name': 'favourite*expression'}

    :param hdf_file: h5py.File object
    :param expression: str expression to be evaluated
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param data_namespace: dict of {'variable name': value} ** note: values will be added to this dict
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param use_stored_data: when True, preferentially takes data from data_namespace, rather than loading from file.
    :return: str expression
    """
    # replace parts of the expression & add attributes to data_namespace
    expression = prepare_expression(hdf_file, expression, hdf_namespace, data_namespace, replace_names)
    # find identifier symbols in expression
    identifiers = find_identifiers(expression)
    if use_stored_data:
        # remove identifiers already in data_namespace so they aren't reloaded
        identifiers = [new_id for new_id in identifiers if new_id not in data_namespace]
    # load identifiers data from file into data_namespace
    generate_namespace(hdf_file, hdf_namespace, identifiers, default, data_namespace)
    logger.info(f"Expression: {expression}\nidentifiers: {identifiers}\n")
    logger.debug(f"hdf data namespace: {data_namespace}\n")
    return expression


def eval_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
             data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
             default: typing.Any = DEFAULT, use_stored_data: bool = False, raise_errors: bool = True) -> typing.Any:
    """
    Evaluate an expression using the namespace of the hdf file

    The following patterns are allowed:
     - 'filename': str, name of hdf_file
     - 'filepath': str, full path of hdf_file
     - '_*name*': str hdf path of *name*
     - '__*name*': str internal name of *name* (e.g. for 'axes')
     - 's_*name*': string representation of dataset (includes units if available)
     - 'd_*name*': return dataset object. **warning**: may result in file not closing on completion
     - '*name*@attr': returns attribute of dataset *name*
     - '*name*?(default)': returns default if *name* doesn't exist
     - '(name1|name2|name3)': returns the first available of the names
     - '(name1|name2?(default))': returns the first available name or default

    Additional variables can be added to the evaluation local namespace using data_namespace.

    Shorthand variables for expressions can be assigned using replace_names = {'new_name': 'favorite*expression'}

    :param hdf_file: h5py.File object
    :param expression: str expression to be evaluated
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param data_namespace: dict of {'variable name': value} (note, this object will be updated)
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param use_stored_data: when True, preferentially takes data from data_namespace, rather than loading from file.
    :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
    :return: asteval(expression)
    """
    if not expression.strip():  # don't evaluate empty strings
        return expression
    # replace names with expressions
    expression = replace_expression_vars(expression, replace_names)
    # if expression is a hdf path, just return the data
    if expression in hdf_file:
        return dataset2data(hdf_file[expression])
    expression = prepare_expression_load_data(hdf_file, expression, hdf_namespace, data_namespace,
                                              replace_names, default, use_stored_data)
    logger.debug(f"evaluating expression: '{expression}'")
    # evaluate expression within namespace
    safe_eval = asteval.Interpreter(user_symbols=data_namespace, use_numpy=True)
    result = safe_eval(expression, raise_errors=raise_errors)
    if safe_eval.error_msg:
        logger.error(f"Expression: {expression} gives error message: {safe_eval.error_msg}")
        return f"Error: '{safe_eval.error_msg}'"
    return result


def format_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
               data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
               default: typing.Any = DEFAULT, use_stored_data: bool = False, raise_errors: bool = True) -> str:
    """
    Evaluate a formatted string expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression using {name} format specifiers
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param data_namespace: dict of {'variable name': value}
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param use_stored_data: when True, preferentially takes data from data_namespace, rather than loading from file.
    :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
    :return: eval_hdf(f"expression")
    """
    expression = 'f"""' + expression + '"""'  # convert to fstr
    return eval_hdf(
        hdf_file=hdf_file,
        expression=expression,
        hdf_namespace=hdf_namespace,
        data_namespace=data_namespace,
        replace_names=replace_names,
        default=default,
        use_stored_data=use_stored_data,
        raise_errors=raise_errors
    )


