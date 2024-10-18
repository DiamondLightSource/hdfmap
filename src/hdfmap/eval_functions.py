"""
hdf eval functions
"""

import sys
import os
import ast
import builtins
import datetime
import re
import typing
import numpy as np
import h5py

from .logging import create_logger

# parameters
GLOBALS = {'np': np}
GLOBALS_NAMELIST = dir(builtins) + list(GLOBALS.keys())
DEFAULT: typing.Any = np.array('--')  # default return in eval
logger = create_logger(__name__)
# regex patterns
special_characters = re.compile(r'\W')  # finds all special non-alphanumberic characters
long_floats = re.compile(r'\d+\.\d{5,}')  # finds floats with long trailing decimals
# fromisoformat requires python 3.11+
datetime_converter = np.vectorize(lambda x: datetime.datetime.fromisoformat(x.decode() if hasattr(x, 'decode') else x))

if sys.version_info < (3, 11, 0):
    logger.warning("Nexus timestamps are not convertable by datetime.fromisoformat in python version <3.11")


def expression_safe_name(string: str, replace: str = '_') -> str:
    """
    Returns an expression safe name
    :param string: any string
    :param replace: str replace special characters with this
    :return: string with special characters replaced
    """
    return special_characters.sub(replace, string)


def round_string_floats(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    def subfun(m):
        return str(round(float(m.group()), 3))
    return long_floats.sub(subfun, string)


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
    except ValueError:
        try:
            string_dataset = dataset.asstr()[()]
            logger.debug(f"Dataset {repr(dataset)} is string")
            if dataset.ndim == 0:
                return round_string_floats(string_dataset)  # bytes or str -> str
            return string_dataset  # str array
        except ValueError:
            logger.debug(f"Dataset {repr(dataset)} is an unexpected type")
            return np.squeeze(dataset[index])  # other np.ndarray


def dataset2str(dataset: h5py.Dataset, index: int | slice = ()) -> str:
    """
    Read the data from a h5py Dataset and convert to a representative string

     - Strings are given with quotes
     - long floats are shorted by attribute 'decimals'
     - numeric arrays are summarised as "dtype (shape)"
     - string arrays are summarised as "['str0', ...]
     - trailing floats within strings are shortened

    :param dataset: h5py.Dataset containing data
    :param index: index of array (not used if dataset is string/ bytes type)
    :return str: string representation of data
    """
    if np.issubdtype(dataset, np.number):
        logger.debug(f"Dataset {repr(dataset)} is numeric")
        if dataset.size > 1:
            return f"{dataset.dtype} {dataset.shape}"
        value = np.squeeze(dataset[index])  # numeric np.ndarray
        if 'decimals' in dataset.attrs:
            value = value.round(dataset.attrs['decimals'])
        return str(value)
    try:
        # timestamp -> datetime64 -> datetime
        # timestamp = np.squeeze(dataset[index]).astype(np.datetime64).astype(datetime.datetime)
        timestamp = np.squeeze(datetime_converter(dataset))
        logger.debug(f"Dataset {repr(dataset)} is timestamp")
        # single datetime obj vs array of datetime obj
        return f"'{timestamp[()]}'" if timestamp.ndim == 0 else f"['{timestamp[0]}', ...({len(timestamp)})]"
    except ValueError:
        try:
            string_dataset = dataset.asstr()[()]
            logger.debug(f"Dataset {repr(dataset)} is string")
            if dataset.ndim == 0:
                return f"'{round_string_floats(string_dataset)}'"  # bytes or str -> str
            return f"['{string_dataset[0]}', ...({len(string_dataset)})]"  # str array
        except ValueError:
            logger.debug(f"Dataset {repr(dataset)} is an unexpected type")
            return str(np.squeeze(dataset[index]))  # other np.ndarray


def check_unsafe_eval(eval_str: str) -> None:
    """
    Check str for naughty eval arguments such as sys, os or import
    This is not foolproof.
    :param eval_str: str
    :return: pass or raise error
    """
    bad_names = ['import', 'os.', 'sys.', 'open(', 'eval', 'exec']
    for bad in bad_names:
        if bad in eval_str:
            raise Exception('This operation is not allowed as it contains: "%s"' % bad)


def find_identifiers(expression: str) -> list[str]:
    """Returns list of variable names in expression, ommiting builtins and globals"""
    # varnames = re.findall(r'[a-zA-Z]\w*', expression)
    return [node.id for node in ast.walk(ast.parse(expression, mode='eval'))
            if type(node) is ast.Name and node.id not in GLOBALS_NAMELIST]


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

    :param hdf_file: h5py.File object
    :param hdf_namespace: locations of data in hdf file, dict[identifier]='/hdf/dataset/path'
    :param identifiers: list of names to load from hdf_file, if None, use generate all items in name_path
    :param default: any, if varname not in name_path - return default instead
    :return: dict {'name': value, '_name': '/hdf/path'}
    """
    if identifiers is None:
        identifiers = list(hdf_namespace.keys())
    # TODO: add ROI commands e.g. nroi[1,2,3,4] -> default_image([1,2,3,4])
    # TODO: add name@attribute e.g. incident_energy@units -> 'eV'
    # TODO: add name.label e.g. axes.label -> 'eta [Deg]'
    # TODO: add class_name e.g. NXdetector_data
    namespace = {
        name: dataset2data(hdf_file[hdf_namespace[name]])
        for name in identifiers if name in hdf_namespace and hdf_namespace[name] in hdf_file
    }
    defaults = {
        name: default
        for name in identifiers if (name not in hdf_namespace) or (hdf_namespace[name] not in hdf_file)
    }
    hdf_paths = {name: hdf_namespace[name[1:]] for name in identifiers
                 if name.startswith('_') and name[1:] in hdf_namespace}
    # add extra params
    extras = extra_hdf_data(hdf_file)
    return {**defaults, **extras, **hdf_paths, **namespace}


def eval_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
             default: typing.Any = DEFAULT) -> typing.Any:
    """
    Evaluate an expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression to be evaluated
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param default: returned if varname not in namespace
    :return: eval(expression)
    """
    if expression in hdf_file:
        return dataset2data(hdf_file[expression])
    check_unsafe_eval(expression)
    # find identifiers matching names in the namespace
    identifiers = [name for name in hdf_namespace if name in special_characters.split(expression)]
    # find other non-builtin identifiers
    identifiers += [name for name in find_identifiers(expression) if name not in identifiers]
    namespace = generate_namespace(hdf_file, hdf_namespace, identifiers, default)
    logger.info(f"Expression: {expression}\nidentifiers: {identifiers}\n")
    logger.debug(f"namespace: {namespace}\n")
    return eval(expression, GLOBALS, namespace)


def format_hdf(hdf_file: h5py.File, expression: str, hdf_namespace: dict[str, str],
               default: typing.Any = DEFAULT) -> str:
    """
    Evaluate a formatted string expression using the namespace of the hdf file
    :param hdf_file: h5py.File object
    :param expression: str expression using {name} format specifiers
    :param hdf_namespace: dict of {'variable name': '/hdf/dataset/path'}
    :param default: returned if varname not in namespace
    :return: eval_hdf(f"expression")
    """
    expression = 'f"""' + expression + '"""'  # convert to fstr
    return eval_hdf(hdf_file, expression, hdf_namespace, default)


