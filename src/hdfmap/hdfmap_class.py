"""
hdfmap class definition
"""

import h5py

from .eval_functions import eval_hdf, format_hdf

try:
    import hdf5plugin  # required for compressed data
except ImportError:
    print('Warning: hdf5plugin not available.')

# parameters
SEP = '/'  # HDF address separator


def address_name(address: str | bytes) -> str:
    """Convert hdf address to name"""
    if hasattr(address, 'decode'):  # Byte string
        address = address.decode('ascii')
    address = address.replace('.', '_')  # remove dots as cant be evaluated
    name = address.split(SEP)[-1]
    return address.split(SEP)[-1] if name == 'value' else name


def _disp_dict(mydict: dict, indent: int = 10) -> str:
    return '\n'.join([f"{key:>indent}: {value}" for key, value in mydict.items()])


class HdfMap:
    """
    HdfMap object, container for addresses of different objects in an HDF file
        map = HdfMap()
        with h5py.File('file.hdf') as hdf:
            map.populate(hdf)
        size = map.most_common_size()
        map.generate_scannables(size)

    Objects within the HDF file are separated into Groups and Datasets. Each object has a
    defined 'address' and 'name paramater, as well as other attributes
        address -> '/entry/measurement/data' -> the location of an object within the file
        name -> 'data' -> an address expressed as a simple variable name
    Address are unique location within the file but can be used to identify similar objects in other files
    Names may not be unique within a file and generated from the address.

    Names of different types of datasets are stored for arrays (size > 0) and values (size 0)
    Names for scannables relate to all arrays of a particular size
    A combined list of names is provided where scannables > arrays > values

    Attributes:
        map.groups      stores attributes of each group by address
        map.classes     stores list of group addresses by nx_class
        map.datasets    stores attributes of each dataset by address
        map.arrays      stores array dataset addresses by name
        map.values      stores value dataset addresses by name
        map.scannables  stores array dataset addresses with given size, by name
        map.combined    stores array and value addresses (arrays overwrite values)
        map.image_data  stores dataset addresses of image data
    E.G.
        map.groups = {'/hdf/group': ('class', 'name')}
        map.classes = {'class_name': ['/hdf/group1', '/hdf/group2']}
        map.datasets = {'/hdf/group/dataset': ('name', size, shape, {attrs})}
        map.arrays = {'name': '/hdf/group/dataset'}
        map.values = {'name': '/hdf/group/dataset'}
        map.scannables = {'name': '/hdf/group/dataset'}
        map.image_data = {'name': '/hdf/group/dataset'}

    Methods:
        map.populate(h5py.File) -> populates the dictionaries using the  given file
        map.generate_scannables(array_size) -> populates scannables namespace with arrays of same size
        map.most_common_size -> returns the most common dataset size > 1
        map.get_size('name_or_address') -> return dataset size
        map.get_shape('name_or_address') -> return dataset size
        map.get_attr('name_or_address', 'attr') -> return value of dataset attribute
        map.get_class('class name') -> return address of group with class
        map.get_class_datasets('class name') -> return list of dataset addresses in class
    """
    _debug_logger = print

    def __init__(self):
        self._debug = False
        self.filename = ''
        self.groups = {}  # stores attributes of each group by address
        self.classes = {}  # stores group addresses by nx_class
        self.datasets = {}  # stores attributes of each dataset by address
        self.arrays = {}  # stores array dataset addresses by name
        self.values = {}  # stores value dataset addresses by name
        self.scannables = {}  # stores array dataset addresses with given size, by name
        self.combined = {}  # stores array and value addresses (arrays overwrite values)
        self.image_data = {}  # stores dataset addresses of image data

    def __repr__(self):
        return f"HdfMap based on '{self.filename}'"

    def info_groups(self):
        """Return str info on groups"""
        out = f"{repr(self)}\n"
        out += "Groups:\n"
        out += _disp_dict(self.groups, 10)
        out += 'Classes:\n'
        out += _disp_dict(self.classes, 10)
        return out

    def info_datasets(self):
        """Return str info on datasets"""
        out = f"{repr(self)}\n"
        out += "Datasets:\n"
        out += _disp_dict(self.datasets, 10)
        return out

    def info_dataset_types(self):
        """Return str info on dataset types"""
        out = "Values:\n"
        out += _disp_dict(self.values, 10)
        out += "Arrays:\n"
        out += '\n'.join([
            f"{name:>10}: {str(self.datasets[address][2]):10} : {address:60}"
            for name, address in self.arrays.items()
        ])
        out += "Images:\n"
        out += '\n'.join([
            f"{name:>10}: {str(self.datasets[address][2]):10} : {address:60}"
            for name, address in self.image_data.items()
        ])
        return out

    def info_names(self):
        """Return str info on combined namespace"""
        out = "Combined Namespace:\n"
        out += '\n'.join([
            f"{name:>10}: {str(self.datasets[address][2]):10} : {address:60}"
            for name, address in self.combined.items()
        ])
        return out

    def info_scannables(self):
        """Return str info on scannables namespace"""
        out = "Scannables Namespace:\n"
        out += '\n'.join([
            f"{name:>10}: {str(self.datasets[address][2]):10} : {address:60}"
            for name, address in self.scannables.items()
        ])
        return out

    def __str__(self):
        return f"{repr(self)}\n{self.info_names()}\n"

    def debug(self, state=True, logger_function=None):
        """Turn debugging on"""
        self._debug = state
        if logger_function:
            self._debug_logger = logger_function

    def _load_defaults(self, hdf_file: h5py.File):
        """Overloaded method, called before _populate"""
        pass

    def _populate(self, hdf_group: h5py.Group, top_address: str = '',
                  recursive: bool = True, groups: None | list[str] = None):
        """
        populate HdfMap dictionary's using recursive method
        :param hdf_group: HDF group object, from HDF File
        :param top_address: str address of hdf Group, used to build dataset addresses
        :param recursive: if True, will recursively search through subgroups
        :param groups: if not None, will only search subgroups named in list, e.g. ['entry','NX_DATA']
        :return: None
        """
        for key in hdf_group:
            obj = hdf_group.get(key)
            link = hdf_group.get(key, getlink=True)
            address = top_address + SEP + key  # build hdf address - a cross-file unique identifier
            name = address_name(address)
            altname = address_name(obj.attrs['local_name']) if 'local_name' in obj.attrs else name
            if self._debug:
                self._debug_logger(f"{address}  {name}, altname={altname}, link={repr(link)}")

            # Group
            if isinstance(obj, h5py.Group):
                try:
                    nx_class = obj.attrs['NX_class'].decode() if 'NX_class' in obj.attrs else 'Group'
                except AttributeError:
                    nx_class = obj.attrs['NX_class']
                except OSError:
                    nx_class = 'Group'  # if object doesn't have attrs
                self.groups[address] = (nx_class, name)
                if nx_class not in self.classes:
                    self.classes[nx_class] = [address]
                else:
                    self.classes[nx_class].append(address)
                if self._debug:
                    self._debug_logger(f"{address}  HDFGroup: {nx_class}")
                if recursive and (key in groups or nx_class in groups if groups else True):
                    self._populate(obj, address, recursive)

            # Dataset
            elif isinstance(obj, h5py.Dataset) and not isinstance(link, h5py.SoftLink):
                self.datasets[address] = (name, obj.size, obj.shape, dict(obj.attrs))
                if obj.ndim >= 3:
                    self.image_data[address] = (name, obj.size, obj.shape, dict(obj.attrs))
                    self.arrays[name] = address
                    self.arrays[altname] = address
                    if self._debug:
                        self._debug_logger(f"{address}  HDFDataset: image_data & array {name, obj.size, obj.shape}")
                elif obj.ndim > 0:
                    self.arrays[name] = address
                    self.arrays[altname] = address
                    if self._debug:
                        self._debug_logger(f"{address}  HDFDataset: array {name, obj.size, obj.shape}")
                else:
                    self.values[name] = address
                    self.values[altname] = address
                    if self._debug:
                        self._debug_logger(f"{address}  HDFDataset: value")

    def populate(self, hdf_file: h5py.File):
        """Populate all datasets from file"""
        self.filename = hdf_file.filename
        self._load_defaults(hdf_file)
        self._populate(hdf_file)

    def generate_scannables_from_group(self, hdf_group: h5py.Group):
        """Generate scannables list from a specific group, using the first item to define array size"""
        first_dataset = hdf_group[next(iter(hdf_group.keys()))]
        array_size = first_dataset.size
        self.scannables = {k: hdf_group[k].name for k in hdf_group if hdf_group[k].size == array_size}
        self.generate_combined()

    def generate_combined(self):
        self.combined = {**self.values, **self.arrays, **self.scannables}

    def most_common_size(self) -> int:
        """Return most common array size > 1"""
        array_sizes = [
            self.datasets[address][1]
            for name, address in self.arrays.items()
            if self.datasets[address][1] > 1
        ]
        return max(set(array_sizes), key=array_sizes.count)

    def most_common_shape(self) -> tuple:
        """Return most common non-singular array shape"""
        array_shapes = [
            self.datasets[address][2]
            for name, address in self.arrays.items()
            if len(self.datasets[address][2]) > 0
        ]
        return max(set(array_shapes), key=array_shapes.count)

    def generate_scannables(self, array_size) -> None:
        """Populate self.scannables field with datasets size that match array_size"""
        self.scannables = {k: v for k, v in self.arrays.items() if self.datasets[v][1] == array_size}
        # create combined dict, scannables and arrays overwrite values with same name
        self.generate_combined()

    def _get_dataset(self, name_or_address: str, idx: int):
        """Return attribute of dataset"""
        if name_or_address in self.datasets:
            return self.datasets[name_or_address][idx]
        if name_or_address in self.combined:
            return self.datasets[self.combined[name_or_address]][idx]

    def get_address(self, name_or_address):
        """Return address of object in HdfMap"""
        if name_or_address in self.datasets or name_or_address in self.groups:
            return name_or_address
        if name_or_address in self.combined:
            return self.combined[name_or_address]
        if name_or_address in self.classes:
            return self.classes[name_or_address]

    def get_size(self, name_or_address: str) -> int:
        """Return size of dataset"""
        return self._get_dataset(name_or_address, 1)

    def get_shape(self, name_or_address: str) -> tuple:
        """Return shape of dataset"""
        return self._get_dataset(name_or_address, 2)

    def get_attrs(self, name_or_address: str) -> dict:
        """Return attributes of dataset"""
        return self._get_dataset(name_or_address, 3)

    def get_attr(self, name_or_address: str, attr_label: str, default: str = '') -> str:
        """Return named attribute from dataset, or default"""
        attrs = self.get_attrs(name_or_address)
        if attr_label in attrs:
            return attrs[attr_label]
        return default

    def get_class(self, nx_class: str) -> str | None:
        """Return HDF address of first group with nx_class attribute"""
        if nx_class in self.classes:
            return self.classes[nx_class][0]

    def get_class_datasets(self, nx_class: str) -> list[str] | None:
        """Return list of HDF dataset addresses from first group with nx_class attribute"""
        class_address = self.get_class(nx_class)
        if class_address:
            return [address for address in self.datasets if address.startswith(class_address)]

    "--------------------------------------------------------"
    "---------------------- FILE READERS --------------------"
    "--------------------------------------------------------"

    def get_data(self, hdf_file: h5py.File, name_or_address: str, index=(), default=None):
        """Return data from dataset in file"""
        address = self.get_address(name_or_address)
        if address:
            return hdf_file[address][index]
        return default

    def eval(self, hdf_file: h5py.File, expression: str):
        """
        Evaluate an expression using the namespace of the hdf file
        :param hdf_file: h5py.File object
        :param expression: str expression to be evaluated
        :return: eval(expression)
        """
        return eval_hdf(hdf_file, expression, self.combined, self._debug)

    def format_hdf(self, hdf_file: h5py.File, expression: str) -> str:
        """
        Evaluate a formatted string expression using the namespace of the hdf file
        :param hdf_file: h5py.File object
        :param expression: str expression using {name} format specifiers
        :return: eval_hdf(f"expression")
        """
        return format_hdf(hdf_file, expression, self.combined, self._debug)

