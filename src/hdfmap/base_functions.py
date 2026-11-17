import datetime
import re
from types import EllipsisType

import h5py
import numpy as np

from hdfmap.objects import Dataset
from .logging import create_logger

logger = create_logger(__name__)
SEP = '/'  # HDF path separator
OMIT = ['/value', '/data']  # omit these names in paths when determining identifier
LOCAL_NAME = 'local_name'  # dataset attribute name for alt_name (DLS specific)
re_special_characters = re.compile(r'\W')  # finds all special non-alphanumberic characters
datetime_converter = np.vectorize(lambda x: datetime.datetime.fromisoformat(x.decode() if hasattr(x, 'decode') else x))
re_long_floats = re.compile(r'\d+\.\d{5,}')  # finds floats with long trailing decimals


def generate_alt_name(hdf_dataset: h5py.Dataset) -> str | None:
    """Generate alt_name of dataset if 'local_name' in attributes"""
    if LOCAL_NAME in hdf_dataset.attrs:
        alt_name = hdf_dataset.attrs[LOCAL_NAME]
        if hasattr(alt_name, 'decode'):
            alt_name = alt_name.decode()
        return expression_safe_name(alt_name.split('.')[-1])
    return None


def generate_identifier(hdf_path: str | bytes) -> str:
    """
    Generate a valid python identifier from a hdf dataset path or other string
     - Decodes to ascii
     - omits '/value' and '/data'
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
    for omit in OMIT:
        hdf_path = hdf_path.removesuffix(omit)  # omit 'value' or 'data'
    substring = hdf_path.split(SEP)[-1]
    # remove replication (handles local_names 'name.name' convention)
    substring = '.'.join(dict.fromkeys(substring.split('.')))
    return expression_safe_name(substring)


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


def generate_image_roi_slice(start_i: int, stop_i: int, start_j: int, stop_j: int, step_i: int = 1, step_j: int = 1) -> tuple[EllipsisType, slice, slice]:
    """
    Generate indexing slice for region of interest (ROI)
    """
    return Ellipsis, slice(start_i, stop_i, step_i), slice(start_j, stop_j, step_j)


def make_json_serialisable(obj):
    """Convert h5py/numpy objects into JSON-compatible types."""
    if isinstance(obj, bytes):
        return obj.decode()

    if isinstance(obj, np.ndarray):
        return [make_json_serialisable(v) for v in obj.tolist()]

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: make_json_serialisable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serialisable(v) for v in obj]

    return obj


def attrs2dict(obj: h5py.Group | h5py.Dataset) -> dict:
    """Extract obj.attrs as a serialisable dict"""
    return {name: make_json_serialisable(item) for name, item in obj.attrs.items()}


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
        # note that squeeze returns float if np.float64 is given.
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


def dataset_attribute(dataset: h5py.Dataset | Dataset, attribute: str) -> str:
    """
    Return attribute of dataset
    """
    value = dataset.attrs.get(attribute, '')
    if isinstance(value, bytes):
        return value.decode()
    return value
