import os
import numpy as np

from . import load_hdf, HdfMap, NexusMap
from .logging import create_logger


EXTENSIONS = ['.nxs', '.hdf', '.hdf5', '.h5']
DEFAULT_EXTENSION = EXTENSIONS[0]
logger = create_logger(__name__)


def list_files(folder_directory: str, extension=DEFAULT_EXTENSION) -> list[str]:
    """Return list of files in directory with extension, returning list of full file paths"""
    try:
        return sorted(
            (file.path for file in os.scandir(folder_directory) if file.is_file() and file.name.endswith(extension)),
            key=lambda x: os.path.getmtime(x)
        )
    except FileNotFoundError:
        return []


def as_str_list(string: str | list[str]) -> list[str]:
    """
    Helper function to convert str or list of str to list of str
    :param string: str, byteString, list, array
    :return: list of str
    """
    return list(np.asarray(string, dtype=str).reshape(-1))


def create_hdf_map(hdf_filename: str) -> HdfMap:
    """
    Create a HdfMap from a hdf file
    :param hdf_filename: str filename of hdf file
    :return: HdfMap
    """
    with load_hdf(hdf_filename) as hdf:
        hdf_map = HdfMap(hdf)
    return hdf_map


def create_nexus_map(hdf_filename: str, groups: None | list[str] = None,
                     default_entry_only: bool = False) -> NexusMap:
    """
    Create a HdfMap from a NeXus file, loading default parameters and allowing a reduced, single entry map
    :param hdf_filename: str filename of hdf file
    :param groups: list of groups to collect datasets from
    :param default_entry_only: if True, only the first or default entry will be loaded
    :return: NexusMap
    """
    hdf_map = NexusMap()
    with load_hdf(hdf_filename) as hdf:
        hdf_map.populate(hdf, groups=groups, default_entry_only=default_entry_only)
    return hdf_map


def hdf_data(filenames: str | list[str], name_or_path: str | list[str], hdf_map: HdfMap = None,
             index=(), default=None, fixed_output=False):
    """
    General purpose function to retrieve data from HDF files
    :param filenames: str or list of str - file paths
    :param name_or_path: str or list of str - names or paths of HDF datasets
    :param hdf_map: HdfMap object, or None to generate from first file
    :param index: dataset index or slice
    :param default: value to give if dataset doesn't exist in file
    :param fixed_output: if True, always returns list of list
    :return if single file, single dataset: single value
    :return if multi file or multi dataset: list, len(filenames) or len(name_or_path)
    :return if multi file and multi dataset: list[files: list[names]]
    """
    # cast as 1D arrays
    filenames = as_str_list(filenames)
    name_or_path = as_str_list(name_or_path)
    # generate hdf_map
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        logger.info(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append([hdf_map.get_data(hdf, name, index=index, default=default) for name in name_or_path])
    if fixed_output:
        return out
    if len(filenames) == 1 and len(name_or_path) == 1:
        return out[0][0]
    if len(filenames) == 1 and len(name_or_path) > 1:
        return out[0]
    if len(name_or_path) == 1:
        return [val[0] for val in out]
    return out


def hdf_eval(filenames: str | list[str], expression: str, hdf_map: HdfMap = None, default=None, fixed_output=False):
    """
    Evaluate expression using dataset names
    :param filenames: str or list of str - file paths
    :param expression: str expression to evaluate in each file, e.g. "roi2_sum / Transmission"
    :param hdf_map: HdfMap object, or None to generate from first file
    :param default: value to give if dataset doesn't exist in file
    :param fixed_output: if True, always returns list len(filenames)
    :return if single file: single output
    :return if multi file: list, len(filenames)
    """
    # cast as 1D arrays
    filenames = as_str_list(filenames)
    # generate hdf_map
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        logger.info(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append(hdf_map.eval(hdf, expression, default=default))
    if not fixed_output and len(filenames) == 1:
        return out[0]
    return out


def hdf_format(filenames: str | list[str], expression: str, hdf_map: HdfMap = None, default=None, fixed_output=False):
    """
    Evaluate string format expression using dataset names
    :param filenames: str or list of str - file paths
    :param expression: str expression to evaluate in each file, e.g. "the energy is {en:.2f} keV"
    :param hdf_map: HdfMap object, or None to generate from first file
    :param default: value to give if dataset doesn't exist in file
    :param fixed_output: if True, always returns list len(filenames)
    :return if single file: single output
    :return if multi file: list, len(filenames)
    """
    # cast as 1D arrays
    filenames = as_str_list(filenames)
    # generate hdf_map
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        logger.info(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append(hdf_map.format_hdf(hdf, expression, default=default))
    if not fixed_output and len(filenames) == 1:
        return out[0]
    return out


def hdf_image(filenames: str | list[str], index: slice = None, hdf_map: HdfMap = None, fixed_output=False):
    """
    Evaluate string format expression using dataset names
    :param filenames: str or list of str - file paths
    :param index: index or slice of dataset volume, or None to use middle index
    :param hdf_map: HdfMap object, or None to generate from first file
    :param fixed_output: if True, always returns list len(filenames)
    :return if single file: single output - numpy array
    :return if multi file: list, len(filenames)
    """
    # cast as 1D arrays
    filenames = as_str_list(filenames)
    # generate hdf_map
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        logger.info(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append(hdf_map.get_image(hdf, index=index))
    if not fixed_output and len(filenames) == 1:
        return out[0]
    return out


def nexus_data_block(filenames: str | list[str], hdf_map: HdfMap = None, fixed_output=False):
    """
    Create classic dict like dataloader objects from nexus files
    E.G.
        d = nexus_data_block('filename')
        d.scannable -> array
        d.metadata.filename -> value
        d.keys() -> list of items

    :param filenames: str or list of str - file paths
    :param hdf_map: HdfMap object, or None to generate from first file
    :param fixed_output: if True, always returns list len(filenames)
    :return if single file: single output - dict like DataObject
    :return if multi file: list, len(filenames)
    """
    # cast as 1D arrays
    filenames = as_str_list(filenames)
    # generate hdf_map
    if hdf_map is None:
        hdf_map = create_nexus_map(filenames[0])
    out = []
    for filename in filenames:
        logger.info(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append(hdf_map.get_dataholder(hdf))
    if not fixed_output and len(filenames) == 1:
        return out[0]
    return out


def compare_maps(map1: HdfMap | NexusMap, map2: HdfMap | NexusMap) -> str:
    """
    Compare two HdfMap objects
    """
    missing_in_2 = []
    missing_in_1 = []
    different = []
    same = []
    for name1, path1 in map1.combined.items():
        if name1 in map2.combined:
            path2 = map2.combined[name1]
            if path2 != path1:
                different.append(f"{name1}: {path1} != {path2}")
            dataset1 = map1.datasets[path1]
            dataset2 = map2.datasets[path2]
            if dataset1.shape != dataset2.shape:
                different.append(f"{name1}: {dataset1.shape}, {dataset2.shape}")
            else:
                same.append(f"{name1}: {dataset1.shape} : {path1}, {path2}")
        else:
            missing_in_2.append(f"{name1}: {path1}")

    for name2, path2 in map2.combined.items():
        if name2 not in map1.combined:
            missing_in_1.append(f"{name2}: {path2}")

    output = f"Comparing:\n  {map1.filename}, with\n  {map2.filename}\n\n"
    output += "Different items:\n  " + '\n  '.join(different)
    output += f"\n\nMissing in {map1.filename}:\n  " + '\n  '.join(missing_in_1)
    output += f"\n\nMissing in {map2.filename}:\n  " + '\n  '.join(missing_in_2)
    output += '\n'
    return output
