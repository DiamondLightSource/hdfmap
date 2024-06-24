"""
Reloader class
"""

import numpy as np

from .hdfmap_class import HdfMap
from .file_functions import load_hdf, create_hdf_map


class HdfReloader:
    """
    Hdf Reloader
        hdf = HdfReloader('file.hdf')
        [data1, data2] = hdf.get_data(['dataset_name_1', 'dataset_name_2'])
        data = hdf.eval('dataset_name_1 * 100 + 2')
        string = hdf.format('my data is {dataset_name_1:.2f}')
    """

    def __init__(self, hdf_filename: str, hdf_map: HdfMap | None = None):
        self.filename = hdf_filename
        if hdf_map is None:
            self.map = create_hdf_map(hdf_filename)
        else:
            self.map = hdf_map

    def __repr__(self):
        return f"HdfReloader('{self.filename}')"

    def __str__(self):
        return f"{repr(self)}\n{str(self.map)}"

    def _load(self):
        return load_hdf(self.filename)

    def get_address(self, name_or_address: str):
        return self.get_address(name_or_address)

    def get_data(self, name_or_address: str | list[str], index=()):
        name_or_address = np.reshape(name_or_address, -1)
        with self._load() as hdf:
            out = [self.map.get_data(hdf, name, index) for name in name_or_address]
        if name_or_address.size == 1:
            return out[0]
        return out

    def eval(self, expression: str):
        with self._load() as hdf:
            return self.map.eval(hdf, expression)

    def format(self, expression: str):
        with self._load() as hdf:
            return self.map.format_hdf(hdf, expression)

