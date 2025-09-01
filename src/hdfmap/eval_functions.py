"""
hdf eval functions
"""

import sys
import os
import ast
import asteval
import datetime
import re
import typing
import numpy as np
import h5py

from .logging import create_logger

# parameters
GLOBALS_NAMELIST = asteval.make_symbol_table(use_numpy=True).keys()
DEFAULT: typing.Any = np.array('--')  # default return in eval
SEP = '/'  # HDF path separator
OMIT = '/value'  # omit this name in paths when determining identifier
logger = create_logger(__name__)
# regex patterns
re_special_characters = re.compile(r'\W')  # finds all special non-alphanumberic characters
re_long_floats = re.compile(r'\d+\.\d{5,}')  # finds floats with long trailing decimals
re_dataset_attributes = re.compile(r'([a-zA-Z_]\w*)@([a-zA-Z_]\w*)')  # finds 'name@attribute' in expressions
re_dataset_default = re.compile(r'(\w+)\?\((.+?)\)')  # finds 'name?('noname'), return (name, 'noname')
re_dataset_alternate = re.compile(r'\((\w\S*\|\w\S*?)\)')  # finds '(name1|name2|name3)', return 'name1|name2|name3'
# fromisoformat requires python 3.11+
datetime_converter = np.vectorize(lambda x: datetime.datetime.fromisoformat(x.decode() if hasattr(x, 'decode') else x))

if sys.version_info < (3, 11, 0):
    logger.warning("Nexus timestamps are not convertable by datetime.fromisoformat in python version <3.11")


def generate_identifier(hdf_path: str | bytes) -> str:
    """
    Generate a valid python identifier from a hdf dataset path or other string
     - Decodes to ascii
     - omits '/value'
     - splits by path separator (/) and takes final element
     - converts special characters to '_'
     - removes replication of strings separated by '_'
    E.G.
        /entry/group/motor1 >> "motor1"
        /entry/group/motor/value >> "motor"
        /entry/group/subgroup.motor >> "subgroup_motor"
        motor.motor >> "motor"
    :param hdf_path: str hdf path address
    :return: str expression safe name
    """
    if hasattr(hdf_path, 'decode'):  # Byte string
        hdf_path = hdf_path.decode('ascii')
    if hdf_path.endswith(OMIT):
        hdf_path = hdf_path[:-len(OMIT)]  # omit 'value'
    substrings = hdf_path.split(SEP)
    name = expression_safe_name(substrings[-1])
    # remove replication (handles local_names 'name.name' convention)
    return '_'.join(dict.fromkeys(name.split('_')))


def build_hdf_path(*args: str | bytes) -> str:
    """
    Build path from string or bytes arguments
        '/entry/measurement' = build_hdf_path(b'entry', 'measurement')
    :param args: str or bytes arguments
    :return: str hdf path
    """
    return SEP + SEP.join((arg.decode() if isinstance(arg, bytes) else arg).strip(SEP) for arg in args)


def expression_safe_name(string: str, replace: str = '_') -> str:
    """
    Returns an expression safe name
    :param string: any string
    :param replace: str replace special characters with this
    :return: string with special characters replaced
    """
    return re_special_characters.sub(replace, string)


def round_string_floats(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re_long_floats.sub(subfun, string)


def is_image(shape: tuple[int], min_dim=3):
    """Return True/False if dataset shape is suitable for image data"""
    return len(shape) >= min_dim and (shape[-2] - 1) * (shape[-1] - 1) > 1


def generate_image_roi_slice(start_i: int, stop_i: int, start_j: int, stop_j: int, step_i: int = 1, step_j: int = 1) -> tuple[Ellipsis, slice, slice]:
    """
    Generate indexing slice for region of interest (ROI)
    """
    return Ellipsis, slice(start_i, stop_i, step_i), slice(start_j, stop_j, step_j)


def dataset2data(dataset: h5py.Dataset, index: int | slice = (), direct_load=False) -> datetime.datetime | str | np.ndarray:
    """
    Read the data from a h5py Dataset and convert to either datetime, str or squeezed numpy array

    - numeric size=1 datasets return float
    - numeric size>1 datasets return numpy array
    - string/ byte timestamps (size=1) return datetime.datetime object
    - string/ byte timestamps (size>1) return array of datetime.datetime
    - all other size=1 datasets return string
    - all other size>1 datasets return array of string

    :param dataset: h5py.Dataset containing data
    :param index: index of array (not used if dataset is string/ bytes type)
    :param direct_load: loads the data directly without conversion if True
    :return datetime.datetime: if data is an isoformat string, returns datetime object
    :return str: if data is another string, returns str with long floats rounded
    :return np.ndarray: if data is another numeric object
    """
    if direct_load:
        return dataset[index]
    if np.issubdtype(dataset, np.number):
        logger.debug(f"Dataset {repr(dataset)} is numeric, return numpy array")
        return np.squeeze(dataset[index])  # numeric np.ndarray
    try:
        # str integers will be cast as timestamps (years), capture as int
        return np.squeeze(dataset[index]).astype(int)
    except ValueError:
        pass
    try:
        # timestamp -> np.datetime64 -> datetime, results in Numpy warnings for timezones, wrong time
        # timestamp = np.squeeze(dataset[index]).astype(np.datetime64).astype(datetime.datetime)
        timestamp = np.squeeze(datetime_converter(dataset))
        logger.debug(f"Dataset {repr(dataset)} is timestamp, return array of datetime objects")
        # single datetime obj vs array of datetime obj
        return timestamp[()] if timestamp.ndim == 0 else timestamp
    except (ValueError, OSError):  # OSError arises sometimes for reasons I don't understand (e.g. old I06-1 data)
        try:
            string_dataset = dataset.asstr()[()]
            logger.debug(f"Dataset {repr(dataset)} is string")
            if dataset.ndim == 0:
                return round_string_floats(string_dataset)  # bytes or str -> str
            return string_dataset[index]  # str array
        except ValueError:
            logger.debug(f"Dataset {repr(dataset)} is an unexpected type")
            return np.squeeze(dataset[index])  # other np.ndarray


def dataset2str(dataset: h5py.Dataset, index: int | slice | tuple[slice, ...] = (), units: bool = False) -> str:
    """
    Read the data from a h5py Dataset and convert to a representative string

     - Strings are given with quotes
     - long floats are shorted by attribute 'decimals'
     - numeric arrays are summarised as "dtype (shape)"
     - string arrays are summarised as "['str0', ...]
     - trailing floats within strings are shortened
     - if units=True and units available as attribute, the unit will be appended

    :param dataset: h5py.Dataset containing data
    :param index: index of array (not used if dataset is string/ bytes type)
    :param units: if True and attribute 'units' available, append this to the result
    :return str: string representation of data
    """
    if np.issubdtype(dataset, np.number):
        logger.debug(f"Dataset {repr(dataset)} is numeric")
        if dataset.size > 1:
            return f"{dataset.dtype} {dataset.shape}"
        value = np.squeeze(dataset[index])  # size 1 numeric np.ndarray
        if 'decimals' in dataset.attrs:
            decimals = int(dataset.attrs['decimals'])
            if abs(value) > 1:
                value = value.round(decimals)
            value = np.array2string(value, precision=decimals, separator=', ', floatmode='maxprec')
        if units and 'units' in dataset.attrs:
            value = f"{value} {arg.decode() if isinstance((arg := dataset.attrs['units']), bytes) else arg}"
        return str(value)
    try:
        # timestamp -> datetime64 -> datetime
        # timestamp = np.squeeze(dataset[index]).astype(np.datetime64).astype(datetime.datetime)
        timestamp = np.squeeze(datetime_converter(dataset))
        logger.debug(f"Dataset {repr(dataset)} is timestamp")
        # single datetime obj vs array of datetime obj
        return f"'{timestamp[()]}'" if timestamp.ndim == 0 else f"['{timestamp[0]}', ...({len(timestamp)})]"
    except (ValueError, OSError):  # OSError arises sometimes for reasons I don't understand (e.g. old I06-1 data)
        try:
            string_dataset = dataset.asstr()[()]
            logger.debug(f"Dataset {repr(dataset)} is string")
            if dataset.ndim == 0:
                return f"'{round_string_floats(string_dataset)}'"  # bytes or str -> str
            return f"['{string_dataset[0]}', ...({len(string_dataset)})]"  # str array
        except ValueError:
            logger.debug(f"Dataset {repr(dataset)} is an unexpected type")
            return str(np.squeeze(dataset[index]))  # other np.ndarray


def dataset_attribute(dataset: h5py.Dataset, attribute: str) -> str:
    """
    Return attribute of dataset
    """
    value = dataset.attrs.get(attribute, '')
    if isinstance(value, bytes):
        return value.decode()
    return value


def find_identifiers(expression: str) -> list[str]:
    """Returns list of variable names in expression"""
    return [name for name in asteval.get_ast_names(ast.parse(expression))]


def extra_hdf_data(hdf_file: h5py.File) -> dict:
    """Extract filename, filepath and other additional data fom hdf file"""
    filepath = getattr(hdf_file, 'filename', 'unknown')
    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
    }


def generate_namespace(hdf_file: h5py.File, hdf_namespace: dict[str, str], identifiers: list[str] | None = None,
                       default: typing.Any = DEFAULT) -> dict[str, typing.Any]:
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
    :return: dict {'name': value, '_name': '/hdf/path'}
    """
    if identifiers is None:
        identifiers = list(hdf_namespace.keys())

    def select_ids(startswith=''):
        return (
            (symbol, name) for symbol in identifiers
            if symbol.startswith(startswith) and hdf_namespace.get(name := symbol[len(startswith):], '') in hdf_file
        )

    namespace = {symbol: dataset2data(hdf_file[hdf_namespace[name]]) for symbol, name in select_ids()}
    strings = {symbol: dataset2str(hdf_file[hdf_namespace[name]], units=True) for symbol, name in select_ids('s_')}
    datasets = {symbol: hdf_file[hdf_namespace[name]] for symbol, name in select_ids('d_')}
    hdf_paths = {symbol: hdf_namespace[name] for symbol, name in select_ids('_')}
    hdf_names = {symbol: generate_identifier(hdf_namespace[name]) for symbol, name in select_ids('__')}
    # generate defaults for non-builtin names that are not in the file
    defaults = {
        name: default
        for name in identifiers
        if name not in GLOBALS_NAMELIST and hdf_namespace.get(name, '') not in hdf_file
    }
    return {**defaults, **hdf_paths, **hdf_names, **datasets, **strings, **namespace}


def prepare_expression(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
                       data_namespace: dict[str, typing.Any] | None) -> str:
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
    :return: str expression
    """
    if data_namespace is None:
        data_namespace = {}
    # get extra data
    extra_data = extra_hdf_data(hdf_file)
    # find name@attribute in expression
    attributes = {
        f"attr__{name}_{attr}": dataset_attribute(hdf_file[path], attr)
        for name, attr in re_dataset_attributes.findall(expression)
        if (path := hdf_namespace.get(name, '')) in hdf_file
    }
    extra_data.update(attributes)
    data_namespace.update(extra_data)  # update in the parent function
    # replace name@attribute in expression
    expression = re_dataset_attributes.sub(r'attr__\g<1>_\g<2>', expression)
    # find values with defaults '..?(..)'
    for match in re_dataset_default.finditer(expression):
        name, name_default = match.groups()
        if name not in hdf_namespace:
            expression = expression.replace(match.group(), name_default)
        else:
            expression = expression.replace(match.group(), name)
    # find alternate names '(opt1|opt2|opt3)'
    for alt_names in re_dataset_alternate.findall(expression):  # alt_names = 'opt1|opt2|opt3
        names = alt_names.split('|')
        # first available name in hdf_namespace or last name
        name = next(
            (n for n in names if n in attributes),
            next((n for n in names if n in hdf_namespace), names[-1])
        )
        expression = expression.replace(f"({alt_names})", name)  # replace parentheses
    return expression


def prepare_expression_load_data(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
                                 data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
                                 default: typing.Any = DEFAULT):
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
    :return: str expression
    """
    # replace names with expressions
    for name, replacement in replace_names.items():
        # TODO: make this more reliable by using either regex or ast
        expression = expression.replace(name, replacement)
    # replace parts of the expression
    expression = prepare_expression(hdf_file, expression, hdf_namespace, data_namespace)  # adds data to data_namespace
    # find identifier symbols in expression, but don't reload any in data_namespace
    identifiers = [new_id for new_id in find_identifiers(expression) if new_id not in data_namespace]
    namespace = generate_namespace(hdf_file, hdf_namespace, identifiers, default)
    data_namespace.update(namespace)
    logger.info(f"Expression: {expression}\nidentifiers: {identifiers}\n")
    logger.debug(f"hdf data namespace: {namespace}\n")
    return expression


def eval_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
             data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
             default: typing.Any = DEFAULT, raise_errors: bool = True) -> typing.Any:
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
    :param data_namespace: dict of {'variable name': value}
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
    :return: asteval(expression)
    """
    if not expression.strip():  # don't evaluate empty strings
        return expression
    # replace names with expressions
    for name, replacement in replace_names.items():
        expression = expression.replace(name, replacement)
    # if expression is a hdf path, just return the data
    if expression in hdf_file:
        return dataset2data(hdf_file[expression])
    expression = prepare_expression_load_data(hdf_file, expression, hdf_namespace, data_namespace, replace_names, default)
    # evaluate expression within namespace
    safe_eval = asteval.Interpreter(user_symbols=data_namespace, use_numpy=True)
    result = safe_eval(expression, raise_errors=raise_errors)
    if safe_eval.error_msg:
        logger.error(f"Expression: {expression} gives error message: {safe_eval.error_msg}")
        return f"Error: '{safe_eval.error_msg}'"
    return result


def format_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
               data_namespace: dict[str, typing.Any], replace_names: dict[str, str],
               default: typing.Any = DEFAULT, raise_errors: bool = True) -> str:
    """
    Evaluate a formatted string expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression using {name} format specifiers
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param data_namespace: dict of {'variable name': value}
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
    :return: eval_hdf(f"expression")
    """
    expression = 'f"""' + expression + '"""'  # convert to fstr
    return eval_hdf(hdf_file, expression, hdf_namespace, data_namespace, replace_names, default, raise_errors)


class HdfMapInterpreter(asteval.Interpreter):
    """
    HdfMap implementation of asteval.Interpreter

    Expression is parsed for patterns and loads HDF data before evaluation

        m = HdfMap('file.nxs')
        ii = HdfMapInterpreter(m, replace_names={}, default='', **kwargs)
        out = ii.eval('expression')

    :param hdfmap: HdfMap instance (including hdfmap.filename pointing to the HDF file)
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param kwargs: keyword arguments passed to asteval.Interpreter
    """
    def __init__(self, hdfmap, replace_names: dict[str, str], default: typing.Any = DEFAULT, **kws):
        super().__init__(**kws)
        self.hdfmap = hdfmap
        self.replace_names = replace_names
        self.default_value = default

    def eval(self, expr, lineno=0, show_errors=True, raise_errors=False):
        with self.hdfmap.load_hdf() as hdf:
            new_expression = prepare_expression_load_data(
                hdf_file=hdf,
                expression=expr,
                hdf_namespace=self.hdfmap.combined,
                data_namespace=self.symtable,
                replace_names=self.replace_names,
                default=self.default_value
            )
        return super().eval(new_expression, lineno, show_errors, raise_errors)
