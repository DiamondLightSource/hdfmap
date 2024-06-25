import os
import pathlib
import typing
import h5py
import numpy as np

from .hdfmap_class import HdfMap
from .nexus import NexusMap


EXTENSIONS = ['.nxs', '.hdf', '.hdf5', '.h5']
DEFAULT_EXTENSION = EXTENSIONS[0]
MAX_TEXTVIEW_SIZE = 1000
DEFAULT_ADDRESS = "entry1/scan_command"


def load_hdf(hdf_filename: str) -> h5py.File:
    """Load hdf file, return h5py.File object"""
    return h5py.File(hdf_filename, 'r')


def list_files(folder_directory: str, extension=DEFAULT_EXTENSION) -> list[str]:
    """Return list of files in directory with extension, returning list of full file paths"""
    return sorted(
        (file.path for file in os.scandir(folder_directory) if file.is_file() and file.name.endswith(extension)),
        key=lambda x: os.path.getmtime(x)
    )


def list_path_time_files(directory: str, extension=DEFAULT_EXTENSION) -> list[tuple[str, float, int]]:
    """
    Return list of folders in diectory, along with modified time and number of contained files
        [(path, modified_time(s), nfiles), ...] = list_path_time_files('/folder/path', '.nxs')
    :param directory: directory to look in
    :param extension: file extension to list as nfiles
    :return: [(path, timestamp, nfiles), ...]
    """
    folders = []
    for f in os.scandir(directory):
        if f.is_dir():
            try:
                folders.append((f.path, f.stat().st_mtime, len(list_files(f.path, extension))))
            except PermissionError or FileNotFoundError:
                pass
    return folders


def get_dataset_value(hdf_filename: str, hdf_address: str, default_value: typing.Any = '') -> typing.Any:
    """
    Open HDF file and return value from single dataset
    :param hdf_filename: str filename of hdf file
    :param hdf_address: str hdf address specifier of dataset
    :param default_value: Any - returned value if hdf_address is not available in file
    :return [dataset is array]: str "{type} {shape}"
    :return [dataset is not array]: output of dataset[()]
    :return [dataset doesn't exist]: default_value
    """
    try:
        with load_hdf(hdf_filename) as hdf:
            dataset = hdf.get(hdf_address)
            if isinstance(dataset, h5py.Dataset):
                if dataset.size > 1:
                    return f"{dataset.dtype} {dataset.shape}"
                return dataset[()]
            return default_value
    except Exception:
        return default_value


def get_dataset_str(hdf_filename: str, hdf_address: str) -> str:
    """Generate string describing object in hdf file"""
    with load_hdf(hdf_filename) as hdf:
        obj = hdf.get(hdf_address)
        if obj is None:
            return f"{hdf_address}: doesn't exist in {hdf_filename}"
        try:
            link = repr(hdf.get(hdf_address, getlink=True))
        except RuntimeError:
            link = 'No link'
        myclass = hdf.get(hdf_address, getclass=True)
        out = f"File: {hdf_filename}\nAddress: {hdf_address}\n"
        out += f"name: {obj.name}\n"
        out += f"REPR: {repr(obj)}\n"
        out += f"LINK: {link}\n"
        out += f"Class: {repr(myclass)}\n"
        out += '\nattrs:\n'
        out += '\n'.join([f"  {key:10}: {obj.attrs[key]}" for key in obj.attrs])
        if isinstance(obj, h5py.Dataset):
            out += '\n\n--- Data ---\n'
            out += f"Shape: {obj.shape}\nSize: {obj.size}\nValues:\n"
            if obj.size > MAX_TEXTVIEW_SIZE:
                out += '---To large to view---'
            else:
                out += str(obj[()])
    return out


def search_filename_in_folder(topdir: str, search_str: str = "*.nxs", case_sensitive: bool = False):
    """
    Search recursivley for filenames
    :param topdir: str address of directory to start in
    :param search_str: str to search for, use * to specify unkonwn, e.g. "*.nxs"
    :param case_sensitive:
    :return: list
    """
    return [f.absolute() for f in pathlib.Path(topdir).rglob(search_str, case_sensitive=case_sensitive)]


def search_hdf_files(topdir: str, search_str: str | None = None, extension: str = DEFAULT_EXTENSION,
                     address: str = DEFAULT_ADDRESS, whole_word: bool = False,
                     case_sensitive: bool = False) -> list[str]:
    """
    Search recurslively for hdf files in folder and check within files for dataset
    :param topdir: str address of directory to start in
    :param search_str: str or None, if None, returns any hdf file with this dataset
    :param extension: str extension of files, e.g. ".nxs"
    :param address: str dataset address to check
    :param whole_word: search for whole words only
    :param case_sensitive: search is case-sensitive
    :return: list
    """
    output = []
    search_str = '' if search_str is None else search_str
    search_str = search_str if case_sensitive else search_str.lower()

    for f in pathlib.Path(topdir).rglob(f"*{extension}"):
        if not h5py.is_hdf5(f):
            continue
        with load_hdf(f.name) as hdf:
            dataset = hdf.get(address)
            if dataset:
                if search_str:
                    value = str(dataset[()]) if case_sensitive else str(dataset[()]).lower()
                    if (whole_word and search_str == value) or (search_str in value):
                        output.append(f.name)
                else:
                    output.append(f.name)
    return output


def create_hdf_map(hdf_filename: str, debug: bool = False) -> HdfMap:
    """
    Create a HdfMap from a hdf file
    :param hdf_filename: str filename of hdf file
    :param debug: if True, displays debugging info
    :return: HdfMap
    """
    hdf_map = HdfMap()
    hdf_map.debug(debug)
    with load_hdf(hdf_filename) as hdf:
        hdf_map.populate(hdf)
    size = hdf_map.most_common_size()
    hdf_map.generate_scannables(size)
    return hdf_map


def create_nexus_map(hdf_filename: str, groups: None | list[str] = None, debug: bool = False) -> NexusMap:
    """
    Create a HdfMap from a hdf file
    :param hdf_filename: str filename of hdf file
    :param groups: list of groups to collect datasets from
    :param debug: if True, displays debugging info
    :return: NexusMap
    """
    hdf_map = NexusMap()
    hdf_map.debug(debug)
    with load_hdf(hdf_filename) as hdf:
        hdf_map.populate(hdf, groups=groups)
    return hdf_map


def multifile_get_data(name_or_address: str, filenames: list[str], hdf_map: HdfMap | None = None,
                       index=(), default=None, debug=False):
    """

    :param name_or_address: str
    :param filenames: list of str filenames of hdf files
    :param hdf_map:
    :param index:
    :param default:
    :param debug:
    :return:
    """
    print('Multifile:', filenames)
    filenames = np.reshape(filenames, -1)
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0], debug=debug)
    out = []
    for filename in filenames:
        if debug:
            print(f"\nHDF file: {filename}")
        with load_hdf(filename) as hdf:
            out.append(hdf_map.get_data(hdf, name_or_address, index=index, default=default))
    return out


def multifile_eval(expression: str, filenames: list[str], hdf_map: HdfMap | None = None):
    """

    :param expression:
    :param filenames:
    :param hdf_map:
    :return:
    """
    filenames = np.reshape(filenames, -1)
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        with load_hdf(filename) as hdf:
            out.append(hdf_map.eval(hdf, expression))
    return out


def multifile_format(expression: str, filenames: list[str], hdf_map: HdfMap | None = None):
    """

    :param expression:
    :param filenames:
    :param hdf_map:
    :return:
    """
    filenames = np.reshape(filenames, -1)
    if hdf_map is None:
        hdf_map = create_hdf_map(filenames[0])
    out = []
    for filename in filenames:
        with load_hdf(filename) as hdf:
            out.append(hdf_map.format_hdf(hdf, expression))
    return out
