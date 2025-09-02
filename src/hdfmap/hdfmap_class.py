"""
hdfmap class definition
"""
import typing
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import h5py

from . import load_hdf
from .logging import create_logger
from .eval_functions import (expression_safe_name, extra_hdf_data, eval_hdf, HdfMapInterpreter,
                             format_hdf, dataset2data, dataset2str, is_image,
                             DEFAULT, SEP, generate_identifier, build_hdf_path)


# parameters
LOCAL_NAME = 'local_name'  # dataset attribute name for alt_name
IMAGE_DATA = 'IMAGE'  # namespace name for default image data

# logger
logger = create_logger(__name__)


class Group(typing.NamedTuple):
    nx_class: str
    name: str
    attrs: dict
    datasets: list[str]


class Dataset(typing.NamedTuple):
    name: str
    names: list[str]
    size: int
    shape: tuple[int]
    attrs: dict


def generate_alt_name(hdf_dataset: h5py.Dataset) -> str | None:
    """Generate alt_name of dataset if 'local_name' in attributes"""
    if LOCAL_NAME in hdf_dataset.attrs:
        alt_name = hdf_dataset.attrs[LOCAL_NAME]
        if hasattr(alt_name, 'decode'):
            alt_name = alt_name.decode()
        return expression_safe_name(alt_name.split('.')[-1])
    return None


def disp_dict(mydict: dict, indent: int = 10) -> str:
    return '\n'.join([f"{key:>{indent}}: {value}" for key, value in mydict.items()])


class DataHolder(SimpleNamespace):
    """
    Convert dict to class that looks like a class object with key names as attributes
    Replicates slightly the old scisoftpy.dictutils.DataHolder class, also known as DLS dat format.
        obj = DataHolder(**{'item1': 'value1'})
        obj['item1'] -> 'value1'
        obj.item1 -> 'value1'
    """

    def __getitem__(self, item):
        return self.__dict__.__getitem__(item)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()


class HdfMap:
    """
    HdfMap object, container for paths of different objects in an HDF file

        with h5py.File('file.hdf') as hdf:
            map = HdfMap(hdf)

        map.get_path('data') -> '/entry/measurement/data'
        map['data'] -> '/entry/measurement/data'

        with h5py.File('another_file.hdf') as hdf:
            data = map.get_data(hdf, 'data')
            array = map.get_scannables_array(hdf)
            metadata = map.get_metadata(hdf)
            out = map.eval(hdf, 'data / 10')
            outstr = map.format(hdf, 'the data looks like: {data}')

    Objects within the HDF file are separated into Groups and Datasets. Each object has a
    defined 'path' and 'name' paramater, as well as other attribute:

    - path -> '/entry/measurement/data' -> the location of an object within the file
    - name -> 'data' -> a path expressed as a simple variable name

    Paths are unique location within the file but can be used to identify similar objects in other files
    Names may not be unique within a file and are generated from the final element of the hdf path.

    - When multiple paths produce the same name, the name is overwritten each time, so the last path in the
    file has priority.
    - Names are also stored using the 'local_name' attribute, if it exists

    Names of different types of datasets are stored for arrays (size > 0) and values (size 0)
    Names for scannables relate to all arrays of a particular size
    A combined list of names is provided where scannables > arrays > values

    ### Attributes
    - map.groups      stores attributes of each group by path
    - map.classes     stores list of group paths by nx_class
    - map.datasets    stores attributes of each dataset by path
    - map.arrays      stores array dataset paths by name
    - map.values      stores value dataset paths by name
    - map.metadata   stores value dataset path by altname only
    - map.scannables  stores array dataset paths with given size, by name, all arrays have the same shape
    - map.combined    stores array and value paths (arrays overwrite values)
    - map.image_data  stores dataset paths of image data (arrays with 2+ dimensions or arrays of image files)
    #### E.G.
    - map.groups = {'/hdf/group': ('class', 'name', {attrs}, [datasets])}
    - map.classes = {'class_name': ['/hdf/group1', '/hdf/group2']}
    - map.datasets = {'/hdf/group/dataset': ('name', size, shape, {attrs})}
    - map.arrays = {'name': '/hdf/group/dataset'}
    - map.values = {'name': '/hdf/group/dataset'}
    - map.scannables = {'name': '/hdf/group/dataset'}
    - map.image_data = {'name': '/hdf/group/dataset'}

    ### Methods
    - map.populate(h5py.File) -> populates the dictionaries using the  given file
    - map.generate_scannables(array_size) -> populates scannables namespace with arrays of same size
    - map.most_common_size -> returns the most common dataset size > 1
    - map.get_attr('name_or_path', 'attr') -> return value of dataset attribute
    - map.get_path('name_or_group_or_class') -> returns path of object with name
    - map.get_image_path() -> returns default path of detector dataset (or largest dataset)
    - map.get_group_path('name_or_path_or_class') -> return path of group with class
    - map.get_group_datasets('name_or_path_or_class') -> return list of dataset paths in class
    - map.find_groups(*names_or_classes) -> return list of group paths matching given group names or classes
    - map.find_paths('string') -> return list of dataset paths containing string
    - map.find_names('string') -> return list of dataset names containing string
    - map.find_attr('attr_name') -> return list of paths of groups or datasets containing attribute 'attr_name'
    - map.add_local(local_variable=value) -> add to the local namespace accessed by eval
    - map.add_named_expression(alternate_name='expression') -> add local variables for expressions replaced during eval
    ### File Methods
    - map.get_metadata(h5py.File) -> returns dict of value datasets
    - map.get_scannables(h5py.File) -> returns dict of scannable datasets
    - map.get_scannables_array(h5py.File) -> returns numpy array of scannable datasets
    - map.get_dataholder(h5py.File) -> returns dict like object with metadata and scannables
    - map.get_image(h5py.File, index) -> returns image data (2D float array or str image filename)
    - map.get_data(h5py.File, 'name') -> returns data from dataset
    - map.get_string(h5py.File, 'name') -> returns string summary of dataset
    - map.eval(h5py.File, 'expression') -> returns output of expression
    - map.format(h5py.File, 'string {name}') -> returns output of str expression
    """

    def __init__(self, file: h5py.File | None = None):
        self.filename = ''
        self.all_paths = []
        self.groups = {}  # stores attributes of each group by path
        self.datasets = {}  # stores attributes of each dataset by path
        self.classes = defaultdict(list)  # stores lists of group paths by nx_class
        self.arrays = {}  # stores array dataset paths by name, altname + group_name
        self.values = {}  # stores value dataset paths by name, altname + group_name
        self.metadata = {}  # stores value dataset path by altname only
        self.scannables = {}  # stores array dataset paths with given size, by name
        self.combined = {}  # stores array and value paths (arrays overwrite values)
        self.image_data = {}  # stores dataset paths of image data
        self._local_data = {}  # stores variables and data to be used in eval
        self._alternate_names = {}  # stores variable names for expressions to be evaluated
        self._default_image_path = None

        if isinstance(file, h5py.File):
            self.populate(file)

    def __getitem__(self, item):
        return self.combined[item]

    def __iter__(self):
        return iter(self.combined)

    def __contains__(self, item):
        return item in self.combined or item in self.datasets

    def __call__(self, expression, **kwargs):
        if 'hdf_file' not in kwargs:
            kwargs['hdf_file'] = self.load_hdf()
        return self.eval(expression=expression, **kwargs)

    def __repr__(self):
        return f"HdfMap based on '{self.filename}'"

    def __str__(self):
        out = f"{repr(self)}\n"
        out += self.info_summary()
        out += "\n*use print(self.info_names(combined=True, scannables=True, image_data=True)) to see detail\n"
        return out

    def info_groups(self) -> str:
        """Return str info on groups"""
        out = f"{repr(self)}\n"
        out += "Groups:\n"
        for path, group in self.groups.items():
            out += f"{path} [{group.nx_class}: '{group.name}']\n"
            out += '\n'.join(f"  @{attr}: {self.get_attr(path, attr)}" for attr in group.attrs)
            out += '\n'
            for dataset_name in group.datasets:
                dataset_path = build_hdf_path(path, dataset_name)
                if dataset_path in self.datasets:
                    dataset = self.datasets[dataset_path]
                    out += f"  {dataset_name}: {dataset.shape}\n"
        return out

    def info_classes(self) -> str:
        """Return str info on group class names"""
        out = f"{repr(self)}\n"
        out += 'Classes:\n'
        out += disp_dict(self.classes, 20)
        return out

    def info_datasets(self) -> str:
        """Return str info on datasets"""
        out = f"{repr(self)}\n"
        out += "Datasets:\n"
        out += disp_dict(self.datasets, 20)
        return out

    def info_names(self, arrays=False, values=False, combined=False,
                   metadata=False, scannables=False, image_data=False) -> str:
        """Return str info for different namespaces"""
        if not any((arrays, values, combined, metadata, scannables, image_data)):
            combined = True
        options = [
            ('Arrays', arrays, self.arrays),
            ('Values', values, self.values),
            ('Combined', combined, self.combined),
            ('Metadata', metadata, self.metadata),
            ('Scannables', scannables, self.scannables),
            ('Image Data', image_data, self.image_data),
        ]
        out = ''
        for name, show, namespace in options:
            if show:
                out += f"\n{name} Namespace:\n"
                out += '\n'.join([
                    f"{name:>30}: {str(self.datasets[path].shape):10} : {path:60}"
                    for name, path in namespace.items()
                ])
                out += '\n'
        return out

    def info_summary(self):
        out = [
            "--Paths--",
            f"All paths: {len(self.all_paths)}",
            f"Groups: {len(self.groups)}",
            f"Datasets: {len(self.datasets)}",
            "--Names--",
            f"Classes: {len(self.classes)}",
            f"Arrays: {len(self.arrays)}",
            f"Values: {len(self.values)}",
            f"Combined: {len(self.combined)}",
            f"Metadata: {len(self.metadata)}",
            f"Scannables: {len(self.scannables)}, shape={self.scannables_shape()}, size={self.scannables_length()}",
            f"Image Data: {len(self.image_data)}, shape={self.get_image_shape()}",
        ]
        return '\n'.join(out)

    def _store_class(self, name, path):
        if path not in self.classes[name]:
            self.classes[name].append(path)

    def _store_group(self, hdf_group: h5py.Group, path: str, name: str):

        nx_class = hdf_group.attrs.get('NX_class', default='Group')
        if hasattr(nx_class, 'decode'):
            nx_class = nx_class.decode()
        self.groups[path] = Group(
            nx_class,
            name,
            dict(hdf_group.attrs),
            [key for key, item in hdf_group.items() if isinstance(item, h5py.Dataset)]
        )
        self._store_class(name, path)
        self._store_class(nx_class, path)
        logger.debug(f"{path}  HDFGroup: {nx_class}")
        return nx_class

    def _store_dataset(self, hdf_dataset: h5py.Dataset, hdf_path: str, name: str):
        # New: add group_name to namespace as standard, helps with names like s5/x + s4/x
        # this significantly increases the number of names in namespaces
        group = self.groups[SEP.join(hdf_path.split(SEP)[:-1])]  # group is already stored
        group_name = f"{group.name}_{name}"
        class_name = f"{group.nx_class}_{name}"
        # group_name = generate_identifier(f"{hdf_path.split(SEP)[-2]}_{name}")
        # alt_name = generate_identifier(hdf_dataset.attrs[LOCAL_NAME]) if LOCAL_NAME in hdf_dataset.attrs else None
        alt_name = generate_alt_name(hdf_dataset)
        names = {n: hdf_path for n in {name, group_name, class_name, alt_name} if n}
        self.datasets[hdf_path] = Dataset(
            name=name,
            names=list(names),
            size=hdf_dataset.size,
            shape=hdf_dataset.shape,
            attrs=dict(hdf_dataset.attrs),
        )
        if is_image(hdf_dataset.shape):
            self.image_data[name] = hdf_path
            self.image_data[group_name] = hdf_path
            self.arrays.update(names)
            logger.debug(f"{hdf_path}  HDFDataset: image_data & array {name, hdf_dataset.size, hdf_dataset.shape}")
        elif hdf_dataset.ndim > 0:
            self.arrays.update(names)
            logger.debug(f"{hdf_path}  HDFDataset: array {name, hdf_dataset.size, hdf_dataset.shape}")
        else:
            self.values.update(names)
            if alt_name:
                self.metadata[alt_name] = hdf_path
            logger.debug(f"{hdf_path}  HDFDataset: value")

    def _populate(self, hdf_group: h5py.Group, root: str = '',
                  recursive: bool = True, groups: list[str] = None):
        """
        populate HdfMap dictionary's using recursive method
        :param hdf_group: HDF group object, from HDF File
        :param root: str path of hdf Group, used to build dataset paths
        :param recursive: if True, will recursively search through subgroups
        :param groups: if not None, will only search subgroups named in list, e.g. ['entry','NX_DATA']
        :return: None
        """
        logger.debug(f"{repr(self)}._populate root='{root}'")
        for key in hdf_group:
            obj = hdf_group.get(key)
            link = hdf_group.get(key, getlink=True)
            logger.debug(f"{key}: {repr(obj)} : {repr(link)}")
            if obj is None:
                continue  # dataset may be missing due to a broken link
            hdf_path = root + SEP + key  # build hdf path - a cross-file unique identifier
            # New: store all paths in file, useful for checking if anything was missed, but might be slow
            self.all_paths.append(hdf_path)
            name = generate_identifier(hdf_path)
            logger.debug(f"{hdf_path}:  {name}, link={repr(link)}")

            # Group
            if isinstance(obj, h5py.Group):
                nx_class = self._store_group(obj, hdf_path, name)
                if recursive and (key in groups or nx_class in groups if groups else True):
                    self._populate(obj, hdf_path, recursive)

            # Dataset
            elif isinstance(obj, h5py.Dataset): #18 remove link omission
                self._store_dataset(obj, hdf_path, name)

    def add_local(self, **kwargs):
        """Add value to the local namespace, used in eval"""
        self._local_data.update(kwargs)

    def add_named_expression(self, **kwargs):
        """Add named expression to the local namespace, used in eval"""
        self._alternate_names.update(kwargs)

    def add_roi(self, name: str, cen_i: int | str, cen_j: int | str,
                wid_i: int = 30, wid_j: int = 30, image_name: str = 'IMAGE'):
        """
        Add an image ROI (region of interest) to the named expressions
        The ROI operates on the default IMAGE dataset, loading only the required region from the file.
        The following expressions will be added, for use in self.eval etc.
            *name* -> returns the whole ROI array as a HDF5 dataset
            *name*_total -> returns the sum of each image in the ROI array
            *name*_max -> returns the max of each image in the ROI array
            *name*_min -> returns the min of each image in the ROI array
            *name*_mean -> returns the mean of each image in the ROI array
            *name*_bkg -> returns the background ROI array (area around ROI)
            *name*_rmbkg -> returns the total with background subtracted
            *name*_box -> returns the pixel positions of the ROI
            *name*_bkg_box -> returns the pixel positions of the background ROI

        :param name: string name of the ROI
        :param cen_i: central pixel index along first dimension, can be callable string
        :param cen_j: central pixel index along second dimension, can be callable string
        :param wid_i: full width along first dimension, in pixels
        :param wid_j: full width along second dimension, in pixels
        :param image_name: string name of the image
        """
        wid_i = abs(wid_i) // 2
        wid_j = abs(wid_j) // 2
        islice = f"{cen_i}-{wid_i:.0f} : {cen_i}+{wid_i:.0f}"
        jslice = f"{cen_j}-{wid_j:.0f} : {cen_j}+{wid_j:.0f}"
        dataset = f"d_{image_name}"
        roi_array = dataset + f"[..., {islice}, {jslice}]"
        roi_total = f"{roi_array}.sum(axis=(-1, -2))"
        roi_max = f"{roi_array}.max(axis=(-1, -2))"
        roi_min = f"{roi_array}.min(axis=(-1, -2))"
        roi_mean = f"{roi_array}.mean(axis=(-1, -2))"
        roi_box = (
            'array([' +
            f"[{cen_i}-{wid_i:.0f}, {cen_j}-{wid_j:.0f}]," +
            f"[{cen_i}-{wid_i:.0f}, {cen_j}+{wid_j:.0f}]," +
            f"[{cen_i}+{wid_i:.0f}, {cen_j}+{wid_j:.0f}]," +
            f"[{cen_i}+{wid_i:.0f}, {cen_j}-{wid_j:.0f}]," +
            f"[{cen_i}-{wid_i:.0f}, {cen_j}-{wid_j:.0f}]," +
            '])'
        )

        islice = f"{cen_i}-{wid_i * 2:.0f} : {cen_i}+{wid_i * 2:.0f}"
        jslice = f"{cen_j}-{wid_j * 2:.0f} : {cen_j}+{wid_j * 2:.0f}"
        bkg_array = dataset + f"[..., {islice}, {jslice}]"
        bkg_total = f"{bkg_array}.sum(axis=(-1, -2))"
        roi_bkg_total = f"({bkg_total} - {roi_total})"
        roi_bkg_mean = f"{roi_bkg_total}/(12*{wid_i * wid_j})"
        # Transpose array to broadcast bkg_total
        roi_rmbkg = f"({roi_array}.T - {roi_bkg_mean}).sum(axis=(0, 1))"
        roi_bkg_box = (
            'array([' +
            f"[{cen_i}-{wid_i * 2:.0f}, {cen_j}-{wid_j * 2:.0f}]," +
            f"[{cen_i}-{wid_i * 2:.0f}, {cen_j}+{wid_j * 2:.0f}]," +
            f"[{cen_i}+{wid_i * 2:.0f}, {cen_j}+{wid_j * 2:.0f}]," +
            f"[{cen_i}+{wid_i * 2:.0f}, {cen_j}-{wid_j * 2:.0f}]," +
            f"[{cen_i}-{wid_i * 2:.0f}, {cen_j}-{wid_j * 2:.0f}]," +
            '])'
        )

        alternate_names = {
            f"{name}_total": roi_total,
            f"{name}_max": roi_max,
            f"{name}_min": roi_min,
            f"{name}_mean": roi_mean,
            f"{name}_bkg": roi_bkg_total,
            f"{name}_rmbkg": roi_rmbkg,
            f"{name}_box": roi_box,
            f"{name}_bkg_box": roi_bkg_box,
            name: roi_array,
        }
        self.add_named_expression(**alternate_names)

    def populate(self, hdf_file: h5py.File):
        """Populate all datasets from file"""
        self.filename = hdf_file.filename
        self._local_data.update(extra_hdf_data(hdf_file))
        self._populate(hdf_file)
        size = self.most_common_size()
        self.generate_scannables(size)
        self.generate_combined()

    def generate_combined(self):
        """Finalise the mapped namespace by combining dataset names"""
        # if self.scannables:
        #     # check image datasets are larger than scannables_shape
        #     ndim = len(self.scannables_shape())
        #     self.image_data = {
        #         name: path for name, path in self.image_data.items()
        #         if is_image(self.datasets[path].shape, ndim + 1)
        #     }
        if self.image_data:
            # add default 'image_data'
            self.image_data[IMAGE_DATA] = next(iter(self.image_data.values()))
        self.combined = {**self.values, **self.arrays, **self.image_data, **self.scannables}

    def all_attrs(self) -> dict:
        """Return dict of all attributes in self.datasets and self.groups"""
        ds_attrs = {k: v for path, ds in self.datasets.items() for k, v in ds.attrs.items()}
        grp_attrs = {k: v for path, grp in self.groups.items() for k, v in grp.attrs.items()}
        return {**grp_attrs, **ds_attrs}

    def most_common_size(self) -> int:
        """Return most common array size > 1"""
        array_sizes = [size for name, path in self.arrays.items() if (size := self.datasets[path].size) > 1]
        return max(set(array_sizes), key=array_sizes.count)

    def most_common_shape(self) -> tuple:
        """Return most common non-singular array shape"""
        array_shapes = [shape for name, path in self.arrays.items() if len(shape := self.datasets[path].shape) > 0]
        return max(set(array_shapes), key=array_shapes.count)

    def scannables_length(self) -> int:
        """Return the length of the first axis of scannables array"""
        if not self.scannables:
            return 0
        path = next(iter(self.scannables.values()))
        return self.datasets[path].size

    def scannables_shape(self) -> tuple:
        """Return the shape of the first axis of scannables array"""
        if not self.scannables:
            return (0, )
        path = next(iter(self.scannables.values()))
        return self.datasets[path].shape

    def generate_scannables(self, array_size):
        """Populate self.scannables field with datasets size that match array_size"""
        # self.scannables = {k: v for k, v in self.arrays.items() if self.datasets[v].size == array_size}
        self.scannables = {ds.name: path for path, ds in self.datasets.items() if ds.size == array_size}
        # create combined dict, scannables and arrays overwrite values with same name
        # self.generate_combined()

    def generate_scannables_from_group(self, hdf_group: h5py.Group, group_path: str = None,
                                       dataset_names: list[str] = None):
        """
        Generate scannables list from a specific group, using the first item to define array size
        :param hdf_group: h5py.Group
        :param group_path: str path of group hdf_group if hdf_group.name is incorrect
        :param dataset_names: list of names of group sub-entries to use (use all if None)
        """
        # watch out - hdf_group.name may not point to a location in the file!
        hdf_path = hdf_group.name if group_path is None else group_path
        # list of datasets within group
        if dataset_names:
            dataset_names = [
                name for name in dataset_names if isinstance(hdf_group.get(name), h5py.Dataset)
            ]
        else:
            dataset_names = [name for name, item in hdf_group.items() if isinstance(item, h5py.Dataset)]

        # catch empty groups
        if len(dataset_names) == 0:
            logger.warning(f"HDF Group {hdf_path} has no datasets for scannables")
            self.scannables = {}
        else:
            # use min size dataset as scannable_shape (avoiding image datasets)
            array_size = min(hdf_group[name].size for name in dataset_names)
            self._populate(hdf_group, root=hdf_path, recursive=False)
            self.scannables = {
                name: build_hdf_path(hdf_path, name)
                for name in dataset_names if hdf_group[name].size == array_size  # doesn't check if link
            }
            if len(self.scannables) < 2:
                logger.warning(f"HDF Group {hdf_path} has no consistent datasets for scannables")
                self.scannables = {}
        logger.debug(f"Scannables from group: {list(self.scannables.keys())}")
        # self.generate_combined()

    def generate_scannables_from_names(self, names: list[str]):
        """Generate scannables list from a set of dataset names, using the first item to define array size"""
        # concert names or paths to name (to match alt_name)
        array_names = [n for name in names if (n := generate_identifier(name)) in self.arrays]
        logger.debug(f"Scannables from names: {array_names}")
        array_size = self.datasets[self.arrays[array_names[0]]].size
        self.scannables = {
            name: self.arrays[name] for name in array_names if self.datasets[self.arrays[name]].size == array_size
        }
        # self.generate_combined()

    def first_last_scannables(self, first_names: list[str] = (),
                              last_names: list[str] = ()) -> tuple[dict[str, str], dict[str, str]]:
        """
        Returns default names from scannables
            output first_names returns dict of N names, where N is the number of dimensions in scannable shape
                if fewer axes_names are provided than required, use the first items of scannables instead
            output signal_names returns the last dict item in the list of scannables + signal_names

        :param first_names: list of names of plottable axes in scannables
        :param last_names: list of names of plottable values in scannables
        :return {first_names: path}, {last_names: path}
        """
        all_names = list(first_names) + list(self.scannables.keys()) + list(last_names)
        # check names are in scannables
        warnings = [name for name in all_names if name not in self.scannables]
        all_names = [name for name in all_names if name in self.scannables]
        for name in warnings:
            logger.warning(f"name: '{name}' not in scannables")
        # return correct number of values from start and end
        ndims = len(self.scannables_shape())
        first = {name: self.scannables[name] for name in all_names[:ndims]}
        last = {name: self.scannables[name] for name in all_names[-(len(last_names) or 1):]}
        return first, last

    def get_path(self, name_or_path):
        """Return hdf path of object in HdfMap"""
        if name_or_path in self.datasets or name_or_path in self.groups:
            return name_or_path
        if name_or_path in self.combined:
            return self.combined[name_or_path]
        if name_or_path in self.image_data:
            return self.image_data[name_or_path]
        if name_or_path in self.classes:
            return self.classes[name_or_path][0]  # return first path in list
        return None

    def get_group_path(self, name_or_path):
        """Return group path of object in HdfMap"""
        hdf_path = self.get_path(name_or_path)
        while hdf_path and hdf_path not in self.groups:
            hdf_path = SEP.join(hdf_path.split(SEP)[:-1])
        if not hdf_path:
            return SEP
        return hdf_path

    def get_group_classes(self, name_or_path) -> list[str]:
        """Return list of class names associated with a group or parent group of dataset"""
        group_path = self.get_group_path(name_or_path)
        sub_groups = group_path.split(SEP)
        sub_group_paths = [SEP.join(sub_groups[:n]) for n in range(1, len(sub_groups)+1)]
        sub_group_classes = [self.groups[g].nx_class for g in sub_group_paths if g in self.groups]
        return sub_group_classes

    def get_group_dataset_path(self, group_name, dataset_name) -> str | None:
        """Return path of dataset defined by group and dataset name/attribute"""
        if group_name in self.groups:
            group_paths = [group_name]
        else:
            group_paths = self.classes[group_name]
        for group_path in group_paths:
            group = self.groups[group_path]
            for name in group.datasets:
                dataset_path = build_hdf_path(group_path, name)
                dataset = self.datasets[dataset_path]
                if dataset_name in dataset.names:
                    return dataset_path
        return None

    def find_groups(self, *names_or_classes: str) -> list[str]:
        """
        Find groups that are associated with several names or class names

            [paths, ] = m.find_groups('NXslit', 'NXtransformations', 's1')

        Intended for use finding groups with a certain hierarchy
        :params names_or_classes:  group names or group class names
        :returns: list of hdf group paths, where all groups are associated with all given names or classes.
        """
        # generate a list of all names and class names associated with each group
        # TODO: add all_names to self.generate_combined
        all_names = {p: self.get_group_classes(p) + p.split('/') for p in self.groups}
        return [path for path, names in all_names.items() if all(arg in names for arg in names_or_classes)]

    def find_datasets(self, *names_or_classes: str) -> list[str]:
        """
        Find datasets that are associated with several names or class names

            [paths, ] = m.find_datasets('NXslit', 'x_gap')

        Intended for use finding datasets associated with groups with a certain hierarchy

        Note that arguments are checked against the dataset namespace first, so if the argument appears
        in both lists, it will be assumed to be a dataset.

        :params names_or_classes:  dataset names, group names or group class names
        :returns: list of hdf dataset paths
        """
        args = list(names_or_classes)
        # split args by dataset names
        dataset_names = [args.pop(n) for n, a in enumerate(args) if a in self.combined]
        # find groups from remaining arguments
        group_paths = self.find_groups(*args)
        if not dataset_names:
            # if no datasets are given, return all dataset in group
            return [build_hdf_path(path, name) for path in group_paths for name in self.groups[path].datasets]
        # find all dataset paths associated with name
        dataset_paths = {
            path for name in dataset_names for path in [
                p for p, ds in self.datasets.items() if name in ds.names
            ] + [self.combined[name]] if self.get_group_path(path) in group_paths
        }
        return list(dataset_paths)

    def find_paths(self, string: str, name_only=True, whole_word=False) -> list[str]:
        """
        Find any dataset paths that contain the given string argument

            [paths, ] = m.find_paths('en')  # finds all datasets with name including 'en'

        :param string: str to find in list of datasets
        :param name_only: if True, search only the name of the dataset, not the full path
        :param whole_word: if True, search only for whole-word names (case in-sensitive)
        :return: list of hdf paths
        """
        if whole_word:
            return [path for name, path in self.combined.items() if string.lower() == name.lower()]
        # find string in combined
        combined_paths = {path for name, path in self.combined.items() if string in name}
        if name_only:
            return [
                path for path, dataset in self.datasets.items()
                if string in dataset.name and path not in combined_paths
            ] + list(combined_paths)
        return [
            path for path in self.datasets if string in path and path not in combined_paths
        ] + list(combined_paths)

    def find_names(self, string: str, match_case=False) -> list[str]:
        """
        Find any dataset names that contain the given string argument, searching names in self.combined

            ['m1x', 'm1y', ...] = m.find_names('m1')

        :param string: str to find in list of datasets
        :param match_case: if True, match must be case-sensitive
        :return: list of names
        """
        if match_case:
            return [name for name in self.combined if string in name]
        return [name for name in self.combined if string.lower() in name.lower()]

    def find_attr(self, attr_name: str) -> list[str]:
        """
        Find any dataset or group path with an attribute that contains attr_name.
        :param attr_name: str name of hdfobj.attr
        :return: list of hdf paths
        """
        return [
            path for path, ds in self.datasets.items() if attr_name in ds.attrs
        ] + [
            path for path, grp in self.groups.items() if attr_name in grp.attrs
        ]

    def get_attrs(self, name_or_path: str) -> dict | None:
        """Return attributes of dataset or group"""
        if name_or_path in self.datasets:
            return self.datasets[name_or_path].attrs
        if name_or_path in self.groups:
            return self.groups[name_or_path].attrs
        if name_or_path in self.combined:
            return self.datasets[self.combined[name_or_path]].attrs
        if name_or_path in self.classes:
            return self.groups[self.classes[name_or_path][0]].attrs
        return None

    def get_attr(self, name_or_path: str, attr_label: str, default: str | typing.Any = '') -> str | None:
        """Return named attribute from dataset or group, or default"""
        attrs = self.get_attrs(name_or_path)
        if attrs and attr_label in attrs:
            return attr.decode() if hasattr(attr := attrs[attr_label], 'decode') else attr
        return default

    def set_image_path(self, name_or_path: str):
        """Set the default image path, used by get_image"""
        if name_or_path is None:
            self._default_image_path = None
        else:
            path = self.get_path(name_or_path)
            if path:
                self._default_image_path = path
        logger.info(f"Default image path: {self._default_image_path}")

    def get_image_path(self) -> str:
        """Return HDF path of first dataset in self.image_data"""
        if self._default_image_path:
            return self._default_image_path
        return next(iter(self.image_data.values()), '')

    def get_image_shape(self) -> tuple:
        """Return the scan shape of the detector dataset"""
        path = self.get_image_path()
        if path in self.datasets:
            return self.datasets[path].shape[-2:]
        return 0, 0

    def get_image_index(self, index: int) -> tuple:
        """Return image slice index for index along total scan size"""
        return np.unravel_index(index, self.scannables_shape())

    def get_group_datasets(self, name_or_path: str) -> list[str] | None:
        """Find the path associate with the given name and return all datasets in that group"""
        group_path = self.get_group_path(name_or_path)
        if group_path:
            return self.groups[group_path].datasets
        return None

    "--------------------------------------------------------"
    "---------------------- FILE READERS --------------------"
    "--------------------------------------------------------"

    def load_hdf(self, filename: str | None = None, name_or_path: str = None, **kwargs) -> h5py.File | h5py.Dataset:
        """
        Load hdf file or hdf dataset in open state
        :param filename: str filename of hdf file, or None to use self.filename
        :param name_or_path: if given, returns the dataset
        :param kwargs: additional key-word arguments to pass to h5py.File(...)
        :return: h5py.File object or h5py.dataset object if dataset name given
        """
        if filename is None:
            filename = self.filename
        if name_or_path is None:
            return load_hdf(filename, **kwargs)
        return load_hdf(filename, **kwargs).get(self.get_path(name_or_path))

    def get_data(self, hdf_file: h5py.File, name_or_path: str, index=(), default=None, direct_load=False):
        """
        Return data from dataset in file, converted into either datetime, str or squeezed numpy.array objects
        See hdfmap.eval_functions.dataset2data for more information.
        :param hdf_file: hdf file object
        :param name_or_path: str name or path pointing to dataset in hdf file
        :param index: index or slice of data in hdf file
        :param default: value to return if name not found in hdf file
        :param direct_load: return str, datetime or squeezed array if False, otherwise load data directly
        :return: dataset2data(dataset) -> datetime, str or squeezed array as required.
        """
        path = self.get_path(name_or_path)
        if path and path in hdf_file:
            return dataset2data(hdf_file[path], index, direct_load)
        return default

    def get_string(self, hdf_file: h5py.File, name_or_path: str, index=(), default='', units=False) -> str:
        """
        Return data from dataset in file, converted into string summary of data
        See hdfmap.eval_functions.dataset2str for more information.
        :param hdf_file: hdf file object
        :param name_or_path: str name or path pointing to dataset in hdf file
        :param index: index or slice of data in hdf file
        :param default: value to return if name not found in hdf file
        :param units: if True and attribute 'units' available, append this to the result
        :return: dataset2str(dataset) -> str
        """
        path = self.get_path(name_or_path)
        if path and path in hdf_file:
            return dataset2str(hdf_file[path], index, units=units)
        return default

    def get_metadata(self, hdf_file: h5py.File, default=None, direct_load=False,
                     name_list: list = None, string_output=False) -> dict:
        """
        Return metadata dict from file, loading data for each item in the metadata list
        The metadata list is taken from name_list, otherwise self.metadata or self.values
        :param hdf_file: hdf file object
        :param default: Value to return for names not associated with a dataset
        :param direct_load: if True, loads data from hdf file directory, without conversion
        :param name_list: if available, uses this list of dataset names to generate the metadata list
        :param string_output: if True, returns string summary of each value
        :return: {name: value}
        """
        extra = extra_hdf_data(hdf_file)
        if name_list:
            metadata_paths = {name: self.combined.get(name, '') for name in name_list}
        elif self.metadata:
            metadata_paths = self.metadata
        else:
            logger.warning("'local_names' metadata is not available, using all size=1 datasets.")
            # metadata_paths = self.values
            metadata_paths = {ds.name: path for path, ds in self.datasets.items() if ds.size <= 1}
        if string_output:
            extra = {key: f"'{val}'" for key, val in extra.items()}
            metadata = {
                name: dataset2str(hdf_file[path]) if path in hdf_file else str(default)
                for name, path in metadata_paths.items()
            }
        else:
            metadata = {
                name: dataset2data(hdf_file[path], direct_load=direct_load) if path in hdf_file else default
                for name, path in metadata_paths.items()
            }
        return {**extra, **metadata}

    def create_metadata_list(self, hdf_file: h5py.File, default=None, name_list: list = None,
                             line_separator: str = '\n', value_separator: str = '=') -> str:
        """
        Return a metadata string, using self.get_metadata
        :param hdf_file: hdf file object
        :param default: Value to return for names not associated with a dataset
        :param name_list: if available, uses this list of dataset names to generate the metadata list
        :param line_separator: str separating each metadata parameter
        :param value_separator: str separating name from value
        :return: multi-line string
        """
        return line_separator.join(
            f"{name}{value_separator}{value}"
            for name, value in self.get_metadata(hdf_file, default=default,
                                                 name_list=name_list, string_output=True).items()
        )

    def get_scannables(self, hdf_file: h5py.File, flatten: bool = False, numeric_only: bool = False) -> dict:
        """Return scannables from file (values associated with hdfmap.scannables)"""
        return {
            name: dataset[()].flatten() if flatten else hdf_file[path][()]
            for name, path in self.scannables.items()
            if (dataset := hdf_file.get(path)) and
               (np.issubdtype(dataset.dtype, np.number) if numeric_only else True)
        }

    def get_image(self, hdf_file: h5py.File, index: int | tuple | slice | None = None) -> np.ndarray | None:
        """
        Get image data from file, using default image path
            - If the image path points to a numeric 2+D dataset, returns dataset[index, :, :] -> ndarray
            - If the image path points to a string dataset, returns dataset[index] -> '/path/to/image.tiff'

        Image filenames may be relative to the location of the current file (this is not checked)

        :param hdf_file: hdf file object
        :param index: (slice,) or None to take the middle image
        :return: 2D numpy array of image, or string file path of image
        """
        if index is None:
            index = self.get_image_index(self.scannables_length() // 2)
        if isinstance(index, int):
            index = self.get_image_index(index)
        image_path = self.get_image_path()
        logger.info(f"image path: {image_path}")
        if image_path and image_path in hdf_file:
            # return hdf_file[image_path][index].squeeze()  # remove trailing dimensions
            return self.get_data(hdf_file, image_path, index)  # return array or image paths
        return None

    def _get_numeric_scannables(self, hdf_file: h5py.File) -> list[tuple[str, str, np.ndarray]]:
        """Return numeric scannables available in file"""
        return [
            (name, path, dataset[()].flatten()) for name, path in self.scannables.items()
            if (dataset := hdf_file.get(path)) and np.issubdtype(dataset.dtype, np.number)
        ]

    def get_scannables_array(self, hdf_file: h5py.File, return_structured_array=False) -> np.ndarray:
        """
        Return 2D array of all numeric scannables in file

        :param hdf_file: h5py.File object
        :param return_structured_array: bool, if True, return a Numpy structured array with column headers
        :returns: numpy array with a row for each scannable, shape: (no_scannables, flattened_length)
        """
        _scannables = self._get_numeric_scannables(hdf_file)
        array = np.array([array for name, path, array in _scannables])
        if return_structured_array:
            dtypes = np.dtype([
                (name, hdf_file[path].dtype) for name, path, array in _scannables
            ])
            return np.array([tuple(row) for row in np.transpose(array)], dtype=dtypes)
        return array

    def create_scannables_table(self, hdf_file: h5py.File, delimiter=', ',
                                string_spec='', format_spec='f', default_decimals=8) -> str:
        """
        Return str representation of scannables as a table
        The table starts with a header row given by names of the scannables.
        Each row contains the numeric values for each scannable, formated by the given string spec:
                {value: "string_spec.decimals format_spec"}
            e.g. {value: "5.8f"}
        decimals is taken from each scannables "decimals" attribute if it exits, otherwise uses default
        :param hdf_file: h5py.File object
        :param delimiter: str seperator between each column
        :param string_spec: str first element of float format specifier - length of string
        :param format_spec: str type element of format specifier - 'f'=float, 'e'=exponential, 'g'=general
        :param default_decimals: int default number of decimals given
        :return: str
        """
        _scannables = self._get_numeric_scannables(hdf_file)
        fmt = string_spec + '.%d' + format_spec
        formats = [
            '{:' + fmt % self.get_attr(path, 'decimals', default=default_decimals) + '}'
            for name, path, array in _scannables
        ]

        length = self.scannables_length()
        out = delimiter.join([name for name, _, _ in _scannables]) + '\n'
        out += '\n'.join([
            delimiter.join([
                fmt.format(array[n])
                for (_, path, array), fmt in zip(_scannables, formats)
            ])
            for n in range(length)
        ])
        return out

    def get_dataholder(self, hdf_file: h5py.File, flatten_scannables: bool = False) -> DataHolder:
        """
        Return DataHolder object - a simple replication of scisoftpy.dictutils.DataHolder
        Also known as DLS dat format.
            dataholder.scannable -> array
            dataholder.metadata.value -> metadata
            dataholder['scannable'] -> array
            dataholder.metadata['value'] -> metadata
        :param hdf_file: h5py.File object
        :param flatten_scannables: bool, it True the scannables will be flattened arrays
        :return: data_object (similar to dict)
        """
        metadata = self.get_metadata(hdf_file)
        scannables = self.get_scannables(hdf_file, flatten=flatten_scannables)
        scannables['metadata'] = DataHolder(**metadata)
        return DataHolder(**scannables)

    def eval(self, hdf_file: h5py.File, expression: str, default=DEFAULT, raise_errors: bool = True):
        """
        Evaluate an expression using the namespace of the hdf file
        :param hdf_file: h5py.File object
        :param expression: str expression to be evaluated
        :param default: returned if varname not in namespace
        :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
        :return: eval(expression)
        """
        return eval_hdf(hdf_file, expression, self.combined, self._local_data, self._alternate_names, default, raise_errors)

    def format_hdf(self, hdf_file: h5py.File, expression: str, default=DEFAULT, raise_errors: bool = True) -> str:
        """
        Evaluate a formatted string expression using the namespace of the hdf file
        :param hdf_file: h5py.File object
        :param expression: str expression using {name} format specifiers
        :param default: returned if varname not in namespace
        :param raise_errors: raise exceptions if True, otherwise return str error message as result and log the error
        :return: eval_hdf(f"expression")
        """
        return format_hdf(hdf_file, expression, self.combined, self._local_data, self._alternate_names, default, raise_errors)

    def create_interpreter(self, default=DEFAULT):
        """
        Create an interpreter object for the current file
        The interpreter is a sub-class of asteval.Interpreter that parses expressions for hdfmap eval patters
        and loads data when required.

        The hdf file self.filename is used to extract data and is only opened during evaluation.

            ii = HdfMap.create_interpreter()
            out = ii.eval('expression')
        """
        interpreter = HdfMapInterpreter(
            hdfmap=self,
            replace_names=self._alternate_names,
            default=default,
            user_symbols=self._local_data,
            use_numpy=True
        )
        return interpreter

    def create_dataset_summary(self, hdf_file: h5py.File) -> str:
        """Create summary of all datasets in file"""
        return '\n'.join(f"{path:60}: {self.get_string(hdf_file, path)}" for path in self.datasets)

    def info_data(self, hdf_file: h5py.File) -> str:
        """Return string showing metadata values associated with names"""
        out = repr(self) + '\n'
        out += "Combined Namespace:\n"
        out += '\n'.join([
            f"{name:>30}: " +
            f"{dataset2str(hdf_file[path]):20}" +
            f": {path:60}"
            for name, path in self.combined.items()
        ])
        out += f"\n{self.info_names(scannables=True)}"
        return out
