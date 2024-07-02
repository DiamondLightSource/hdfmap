"""
SRS DataLoader class
 - may be removed
"""

from .hdfmap_class import HdfMap
from .nexus import NexusMap
from .file_functions import load_hdf


class DictObj(dict):
    """
    Convert dict to class that looks like a class object with key names as attributes
        obj = DictObj({'item1': 'value1'}, 'docs')
        obj['item1'] -> 'value1'
        obj.item1 -> 'value1'
        help(obj) -> 'docs'
    """

    def __init__(self, ini_dict: dict, docstr=None):
        # copy dict
        super().__init__(**ini_dict)
        # assign attributes
        for name, value in ini_dict.items():
            setattr(self, name, value)
        # update doc string
        if docstr:
            self.__doc__ = docstr


def create_hdf_datafile(hdf_filename: str, debug: bool = False) -> dict:
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
        metadata = hdf_map.get_metadata(hdf)
        scannables = hdf_map.get_scannables(hdf)
    doc = f"""DataObject for '{hdf_filename}'"""
    scannables['metadata'] = DictObj(metadata, docstr=doc)
    dataobj = DictObj(scannables, docstr=doc)
    return dataobj


def create_nexus_datafile(hdf_filename: str, groups: None | list[str] = None, debug: bool = False) -> dict:
    """
    Create a HdfMap from a hdf file
    :param hdf_filename: str filename of hdf file
    :param debug: if True, displays debugging info
    :return: HdfMap
    """
    hdf_map = NexusMap()
    hdf_map.debug(debug)
    with load_hdf(hdf_filename) as hdf:
        hdf_map.populate(hdf, groups=groups)
        metadata = hdf_map.get_metadata(hdf)
        scannables = hdf_map.get_scannables(hdf)
    doc = f"""DataObject for '{hdf_filename}'"""
    scannables['metadata'] = DictObj(metadata, docstr=doc)
    dataobj = DictObj(scannables, docstr=doc)
    return dataobj

