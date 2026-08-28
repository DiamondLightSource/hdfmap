"""
Reloader class
The ReLoader class is an object that contains a HdfMap and a filename,
allowing the file to be opened only when data is requested.
"""

import os
from typing import Any

import h5py
import numpy as np
from asteval import Interpreter

from . import load_hdf, HdfMap, NexusMap
from .file_functions import create_hdf_map, create_nexus_map
from .eval_functions import DEFAULT, prepare_expression_load_data

Index = int | tuple | slice


class HdfLoader:
    """
    HDF Loader contains the filename and hdfmap for a HDF file, the hdfmap contains all the dataset paths and a
    namespace, allowing data to be called from the file using variable names, loading only the required datasets
    for each operation.

        hdf = HdfLoader('file.hdf')
        [data1, data2] = hdf.get_data(*['dataset_name_1', 'dataset_name_2'])
        data = hdf.eval('dataset_name_1 * 100 + 2')
        string = hdf.format('my data is {dataset_name_1:.2f}')
        print(hdf.summary())

    :param hdf_filename: path to HDF file
    :param hdf_map: HdfMap instance
    """

    def __init__(self, hdf_filename: str, hdf_map: HdfMap | NexusMap | None = None):
        self.filename = hdf_filename
        if hdf_map is None:
            self.map = create_hdf_map(hdf_filename)
        else:
            self.map = hdf_map
        self._prefer_local_data = True
        self._local_data = {
            'filepath': os.path.abspath(hdf_filename),
            'filename': os.path.basename(hdf_filename),
        }

    def __repr__(self):
        return f"HdfReloader('{self.filename}')"

    def __str__(self):
        with self._load() as hdf:
            out = self.map.info_data(hdf)
        return out

    def __getitem__(self, item):
        return self.get_data(item)

    # def __iter__(self):
    #     return iter(self.combined)

    def __contains__(self, item):
        return item in self.map or item in self._local_data

    def __call__(self, expression):
        return self.eval(expression)

    def _load(self) -> h5py.File:
        return load_hdf(self.filename)

    def add_local(self, **kwargs):
        """Add value to the local namespace, used in eval and format"""
        self._local_data.update(kwargs)

    def live_mode(self, live_mode: bool = True):
        """
        Activate the option to reload data from the file each time, rather than from local data

        self.eval('cmd') -> default will load 'cmd' from local storage if available, or from the file
        self.live_mode() -> self.eval('cmd') will return 'cmd' from the file using hdfmap
        self.live_mode(False) -> returns to default behavior
        """
        self._prefer_local_data = live_mode

    def get_hdf_path(self, name_or_path: str) -> str | None:
        """Return hdf path of object in HdfMap"""
        return self.map.get_path(name_or_path)

    def find_hdf_paths(self, string: str, name_only: bool = True, whole_word: bool = False) -> list[str]:
        """
        Find any dataset paths that contain the given string argument
        :param string: str to find in list of datasets
        :param name_only: if True, search only the name of the dataset, not the full path
        :param whole_word: if True, search only for case in-sensitive name
        :return: list of hdf paths
        """
        return self.map.find_paths(string, name_only, whole_word)

    def find_names(self, string: str) -> list[str]:
        """
        Find any dataset names that contain the given string argument, searching names in self.combined
        :param string: str to find in list of datasets
        :return: list of names
        """
        return self.map.find_names(string)

    def get_data(self, *name_or_path, index: Index = (), default=None, direct_load=False):
        """
        Return data from dataset in file, converted into either datetime, str or squeezed numpy.array objects
        See hdfmap.eval_functions.dataset2data for more information.
        :param name_or_path: str name or path pointing to dataset in hdf file
        :param index: index or slice of data in hdf file
        :param default: value to return if name not found in hdf file
        :param direct_load: return str, datetime or squeezed array if False, otherwise load data directly
        :return: dataset2data(dataset) -> datetime, str or squeezed array as required.
        """
        with self._load() as hdf:
            out = [self.map.get_data(hdf, name, index, default, direct_load) for name in name_or_path]
        if len(name_or_path) == 1:
            return out[0]
        return out

    def get_string(self, *name_or_path, index: Index = (), default='', units=False):
        """
        Return data from dataset in file, converted into summary string
        See hdfmap.eval_functions.dataset2data for more information.
        :param name_or_path: str name or path pointing to dataset in hdf file
        :param index: index or slice of data in hdf file
        :param default: value to return if name not found in hdf file
        :param units: if True and attribute 'units' available, append this to the result
        :return: dataset2str(dataset) -> str
        """
        with self._load() as hdf:
            out = [self.map.get_string(hdf, name, index, default, units) for name in name_or_path]
        if len(name_or_path) == 1:
            return out[0]
        return out

    def get_image(self, index: Index | None = None) -> np.ndarray | None:
        """
        Get image data from file, using default image path
        :param index: (slice,) or None to take the middle image
        :return: numpy array of image
        """
        with self._load() as hdf:
            return self.map.get_image(hdf, index)

    def get_metadata(self, defaults=None):
        with self._load() as hdf:
            return self.map.get_metadata(hdf, default=defaults)

    def get_scannables(self):
        """Return scannables from file (values associated with hdfmap.scannables)"""
        with self._load() as hdf:
            return self.map.get_scannables(hdf)

    def summary(self) -> str:
        """Return string summary of datasets"""
        with self._load() as hdf:
            return self.map.create_dataset_summary(hdf)

    def generate_expression(self, expression: str, default=DEFAULT,
                            prefer_local: bool | None = None) -> tuple[str, dict[str, Any]]:
        """
        Evaluate an expression using the namespace of the hdf file,
        returning the evaluated expression and dictionary of data for identifiers

            expression, data = self.generate_eval_expression('signal / (monitor|ic1monitor)')

        This function serves as a useful way to debug expressions for the eval_hdf function.
        Note that the hdf file object must be included as the way the expression is evaluated
        means individual expression components are checked against the HdfMap namespace and
        the hdf file, allowing lazy loading of data (loading only the data needed).

        :param expression: str expression to be evaluated
        :param default: returned if varname not in namespace
        :param prefer_local: uses values in local_data first if available when True
        :return: expression, dict - data namespace
        """
        prefer_local = self._prefer_local_data if prefer_local is None else prefer_local
        if prefer_local and expression in self._local_data:
            return expression, {expression: self._local_data[expression]}
        with self._load() as hdf:
            return self.map.generate_eval_expression(
                hdf_file=hdf,
                expression=expression,
                default=default,
                local_data=self._local_data,
                prefer_local=prefer_local,
            )

    def eval(self, expression: str, default=DEFAULT, prefer_local: bool | None = None, raise_errors: bool = True):
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

        :param expression: str expression to be evaluated
        :param default: returned if varname not in namespace
        :param prefer_local: if True, uses values in local_data first if available
        :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
        :return: eval(expression)
        """
        prefer_local = self._prefer_local_data if prefer_local is None else prefer_local
        if prefer_local and expression in self._local_data:
            return self._local_data[expression]
        with self._load() as hdf:
            return self.map.eval(
                hdf_file=hdf,
                expression=expression,
                default=default,
                local_data=self._local_data,
                prefer_local=prefer_local,
                raise_errors=raise_errors
            )

    def format(self, expression: str, default=DEFAULT, prefer_local: bool | None = None, raise_errors: bool = True) -> str:
        """
        Evaluate a formatted string expression using the namespace of the hdf file
        Identifiers from the namespace can be called inside {} as a
        formatted f-string.

        E.G.
            expression = '{scan_command} E={mean(incident_energy):.2f}'
            output = scan.format(expression)

        :param expression: str expression using {name} format specifiers
        :param default: returned if varname not in namespace
        :param prefer_local: if True, uses values in local_data first if available
        :param raise_errors: raise exceptions if True, otherwise return str error message
        :return: eval_hdf(f"expression")
        """
        with self._load() as hdf:
            return self.map.format_hdf(
                hdf_file=hdf,
                expression=expression,
                default=default,
                local_data=self._local_data,
                prefer_local=self._prefer_local_data if prefer_local is None else prefer_local,
                raise_errors=raise_errors
            )


class NexusLoader(HdfLoader):
    """
    Nexus Loader
    contains the filename and hdfmap for a NeXus file, the hdfmap contains all the dataset paths and a
    namespace, allowing data to be called from the file using variable names, loading only the required datasets
    for each operation.

        hdf = NexusLoader('file.hdf')
        [data1, data2] = hdf.get_data(['dataset_name_1', 'dataset_name_2'])
        data = hdf.eval('dataset_name_1 * 100 + 2')
        string = hdf.format('my data is {dataset_name_1:.2f}')

    :param nxs_filename: path to HDF file
    :param hdf_map: NexusMap instance, or None to generate
    """
    map: NexusMap

    def __init__(self, nxs_filename: str, hdf_map: NexusMap | None = None):
        if not hdf_map:
            hdf_map = create_nexus_map(nxs_filename)
        super().__init__(nxs_filename, hdf_map)

    def get_plot_data(self) -> dict:
        """Return dict of useful plot data"""
        with self._load() as hdf:
            return self.map.get_plot_data(hdf)


class HdfMapInterpreter(Interpreter):
    """
    HdfMap implementation of asteval.Interpreter

    Expression is parsed for patterns and loads HDF data before evaluation.

        ii = HdfMapInterpreter('file.nxs', replace_names={}, default='', **kwargs)
        out = ii.eval('expression')

    :param filename: path to HDF file
    :param hdfmap: HdfMap instance
    :param replace_names: dict of {'variable_name': expression}
    :param default: returned if varname not in namespace
    :param kwargs: keyword arguments passed to asteval.Interpreter
    """
    def __init__(self, filename: str, hdfmap: HdfMap | NexusMap | None = None,
                 replace_names: dict[str, str] | None = None,
                 default: Any = DEFAULT, **kws):
        super().__init__(**kws)
        self.filename = filename
        if hdfmap is None:
            hdfmap = create_nexus_map(filename) if filename.endswith('.nxs') else create_hdf_map(filename)
        self.hdfmap = hdfmap
        self.replace_names: dict[str, str] = replace_names or {}
        self.default_value = default
        self.use_stored_data = False

    def eval(self, expr, lineno=0, show_errors=True, raise_errors=False):
        with load_hdf(self.filename) as hdf:
            new_expression = prepare_expression_load_data(
                hdf_file=hdf,
                expression=expr,
                hdf_namespace=self.hdfmap.combined,
                data_namespace=self.symtable or {},
                replace_names=self.replace_names,
                default=self.default_value,
                use_stored_data=self.use_stored_data
            )
        return super().eval(new_expression, lineno, show_errors, raise_errors)
