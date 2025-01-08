"""
Nexus Related functions and nexus class
"""

import h5py

from .logging import create_logger
from .hdfmap_class import HdfMap, disp_dict
from .eval_functions import generate_identifier, build_hdf_path

NX_CLASS = 'NX_class'
NX_ENTRY = 'NXentry'
NX_DATA = 'NXdata'
NX_LOCALNAME = 'local_name'
NX_DEFAULT = 'default'
NX_RUN = 'entry_identifier'
NX_CMD = 'scan_command'
NX_TITLE = 'title'
NX_MEASUREMENT = 'measurement'
NX_SCANFIELDS = 'scan_fields'
NX_AUXILIARY = 'auxiliary_signals'
NX_SIGNAL = 'signal'
NX_AXES = 'axes'
NX_DETECTOR = 'NXdetector'
NX_DETECTOR_DATA = 'data'
NX_UNITS = 'units'
logger = create_logger(__name__)


def check_nexus_class(hdf_group: h5py.Group, nxclass: str) -> bool:
    """
    Check if hdf_group is a certain NX_class
    :param hdf_group: hdf or nexus group object
    :param nxclass: str name in NX_class attribute
    :return: True/False
    """
    return (hdf_group and
            (group_class := hdf_group.attrs.get(NX_CLASS)) is not None and
            (group_class.decode() if isinstance(group_class, bytes) else group_class) == nxclass)


def default_nxentry(hdf_file: h5py.File) -> str | bytes:
    """
    Return the default NXentry path, or the first NXentry if there is no default, errors if no NXentry

    See: https://manual.nexusformat.org/datarules.html#version-3
    """
    if NX_DEFAULT in hdf_file.attrs and isinstance(hdf_file.get(entry := hdf_file.attrs[NX_DEFAULT]), h5py.Group):
        return entry
    logger.info('File has no default NXEntry, using the first one available')
    return next(path for path in hdf_file if check_nexus_class(hdf_file.get(path), NX_ENTRY))


def default_nxdata(entry_group: h5py.Group) -> str | bytes:
    """
    Return the default NXdata path within an NXentry group

    See: https://manual.nexusformat.org/datarules.html#version-3
    """
    if NX_DEFAULT in entry_group.attrs:
        nx_data_name = entry_group.attrs[NX_DEFAULT]
    else:
        logger.warning(f"No Default NXData group found, using {NX_MEASUREMENT}")
        nx_data_name = NX_MEASUREMENT
    if nx_data_name not in entry_group:
        logger.warning(f"{nx_data_name} not available, using first NXdata group in Entry")
        nx_data_name = next(name for name in entry_group if check_nexus_class(entry_group.get(name), NX_DATA))
        logger.info(f"First NXData group in Entry: {nx_data_name}")
    return nx_data_name


def find_nexus_defaults(hdf_file: h5py.File, nx_data_path: str | None = None) -> tuple[list[str], list[str]]:
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry group in top File group
     - find "default" data group in entry (or 'measurement', or first 'NXdata')
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate paths of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files

    See: https://manual.nexusformat.org/datarules.html#version-3

    :param hdf_file: open HDF file object, i.e. h5py.File(...)
    :param nx_data_path: hdf path of NXdata group, or None to find the default
    :return axes_paths: list of str hdf paths for axes datasets
    :return signal_paths: list of str hdf paths for signal dataset
    """
    # From: https://manual.nexusformat.org/examples/python/plotting/index.html

    if nx_data_path is None:
        # find the default NXentry group
        nx_entry_name = default_nxentry(hdf_file)
        nx_entry = hdf_file[nx_entry_name]
        # find the default NXdata group
        nx_data_name = default_nxdata(nx_entry)
        nx_data_path = build_hdf_path(nx_entry_name, nx_data_name)
    nx_data = hdf_file[nx_data_path]

    # find the axes field(s)
    if isinstance(axes := nx_data.attrs[NX_AXES], (str, bytes)):
        axes_paths = [build_hdf_path(nx_data_path, axes)]
    else:
        axes_paths = [build_hdf_path(nx_data_path, _axes) for _axes in axes]
    # get the signal field
    if NX_SIGNAL in nx_data.attrs:
        if isinstance(signal := nx_data.attrs[NX_SIGNAL], (str, bytes)):
            signal_paths = [build_hdf_path(nx_data_path, signal)]
        else:
            signal_paths = [build_hdf_path(nx_data_path, _signal) for _signal in signal]
    else:
        signal_paths = [build_hdf_path(nx_data_path, NX_DETECTOR_DATA)]
    # Auxiliary signals
    if NX_AUXILIARY in nx_data.attrs:
        signal_paths.extend([
            build_hdf_path(nx_data_path, signal) for signal in nx_data.attrs[NX_AUXILIARY]
        ])
    return axes_paths, signal_paths


def find_nexus_data_strict(hdf_file: h5py.File) -> tuple[list[h5py.Dataset], h5py.Dataset]:
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry group in top File group
     - find "default" data group in entry
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate paths of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_file: open HDF file object, i.e. h5py.File(...)
    :return axes_datasets: list of dataset objects for axes
    :return signal_dataset: dataset object for plot axis
    """
    # From: https://manual.nexusformat.org/examples/python/plotting/index.html
    # find the default NXentry group
    nx_entry = hdf_file[hdf_file.attrs[NX_DEFAULT]]
    # find the default NXdata group
    nx_data = nx_entry[nx_entry.attrs[NX_DEFAULT]]
    # find the axes field(s)
    if isinstance(nx_data.attrs[NX_AXES], (str, bytes)):
        axes_datasets = [nx_data[nx_data.attrs[NX_AXES]]]
    else:
        axes_datasets = [nx_data[_axes] for _axes in nx_data.attrs[NX_AXES]]
    # find the signal field
    signal_dataset = nx_data[nx_data.attrs[NX_SIGNAL]]
    return axes_datasets, signal_dataset


def names_from_scan_fields(hdf_file: h5py.File, scan_fields_path: str) -> list[str]:
    """
    Extract a list of dataset names from the diamond_scan/scan_fields dataset

    scan_fields stores scannables as class_name.dataset_name, return only the dataset_name

    :param hdf_file:
    :param scan_fields_path:
    :returns: ['names',]
    """
    scan_fields_dataset = hdf_file.get(scan_fields_path)
    if scan_fields_dataset:
        return [name.decode().split('.')[-1] for name in scan_fields_dataset[()]]
    return []


class NexusMap(HdfMap):
    """
    HdfMap for Nexus (.nxs) files

    Extends the HdfMap class with additional behaviours for NeXus files.
    http://www.nexusformat.org/

    E.G.
    nxmap = NexusMap()
    with h5py.File('file.nxs', 'r') as nxs:
        nxmap.populate(nxs, default_entry_only=True)  # populates only from the default entry

    # Special behaviour
    nxmap['axes'] -> return path of default axes dataset
    nxmap['signal'] -> return path of default signal dataset
    """

    def __repr__(self):
        return f"NexusMap based on '{self.filename}'"

    def all_nxclasses(self) -> list[str]:
        """Return list of unique NX_class attributes used in NXgroups"""
        return list({
            nxclass.decode() if isinstance(nxclass, bytes) else nxclass
            for path, grp in self.groups.items() if (nxclass := grp.attrs.get(NX_CLASS))
        })

    def info_nexus(self) -> str:
        """Return str info on nexus format"""
        out = f"{repr(self)}\n"
        out += f"{NX_CLASS}:\n"
        nx_classes = self.all_nxclasses()
        out += disp_dict({k: v for k, v in self.classes.items() if k in nx_classes}, 20)
        out += '\nDefaults:\n'
        out += f"  @{NX_DEFAULT}: {self.find_attr(NX_DEFAULT)}\n"
        out += f"  @{NX_AXES}: {self.get_path(NX_AXES)}\n"
        out += f"  @{NX_SIGNAL}: {self.get_path(NX_SIGNAL)}\n"
        return out

    def _default_nexus_paths(self, hdf_file):
        """Load Nexus default axes and signal"""
        try:
            # find the default NXentry group
            nx_entry_name = default_nxentry(hdf_file)
            nx_entry = hdf_file[nx_entry_name]
            nx_entry_path = build_hdf_path(nx_entry_name)
            self._store_group(nx_entry, nx_entry_path, NX_ENTRY)
            # find the default NXdata group
            nx_data_name = default_nxdata(nx_entry)
            nx_data = nx_entry[nx_data_name]
            nx_data_path = build_hdf_path(nx_entry_name, nx_data_name)
            self._store_group(nx_data, nx_data_path, NX_DATA)

            axes_paths, signal_paths = find_nexus_defaults(hdf_file, nx_data_path)
            if axes_paths and isinstance(hdf_file.get(axes_paths[0]), h5py.Dataset):
                self.arrays[NX_AXES] = axes_paths[0]
                n = 0
                for axes_path in axes_paths:
                    if isinstance(hdf_file.get(axes_path), h5py.Dataset):
                        self.arrays[f"{NX_AXES}{n}"] = axes_path
                        n += 1
                logger.info(f"DEFAULT axes: {axes_paths}")
            if signal_paths and isinstance(hdf_file.get(signal_paths[0]), h5py.Dataset):
                self.arrays[NX_SIGNAL] = signal_paths[0]
                n = 0
                for signal_path in signal_paths:
                    if isinstance(hdf_file.get(signal_path), h5py.Dataset):
                        self.arrays[f"{NX_SIGNAL}{n}"] = signal_path
                        n += 1
                logger.info(f"DEFAULT signals: {signal_paths}")
        except KeyError:
            pass

    def nexus_defaults(self) -> tuple[list[str], list[str]]:
        """Return default axes and signal paths"""
        axes_paths = [self.arrays[axes] for n in range(10) if (axes := f"{NX_AXES}{n}") in self.arrays]
        signal_paths = [self.arrays[signal] for n in range(10) if (signal := f"{NX_SIGNAL}{n}") in self.arrays]
        return axes_paths, signal_paths

    def generate_scannables_from_nxdata(self, hdf_file: h5py.File, use_auxiliary: bool = True):
        """Generate scannables from default NXdata, using axuiliary_names if available"""
        # find the default NXdata group and generate the scannables list
        # nx_entry = hdf_file.get(default_nxentry(hdf_file))
        # nx_data = nx_entry.get(default_nxdata(nx_entry))
        nx_entry = hdf_file.get(self.classes[NX_ENTRY][0])  # classes[NX_ENTRY] pre-populated by _default_nexus_paths
        nx_data = hdf_file.get(self.classes[NX_DATA][0])  # classes[NX_DATA] pre-populated by _default_nexus_paths
        logger.info(f"{nx_entry}, {nx_data}")
        if nx_data:
            logger.info(f"Generating Scannables from NXData: {nx_data.name}")
            if use_auxiliary and NX_AUXILIARY in nx_data.attrs:
                signals = list(nx_data.attrs[NX_AUXILIARY])
                if NX_SIGNAL in nx_data.attrs:
                    signals.insert(0, nx_data.attrs[NX_SIGNAL])
                signals = [i.decode() if isinstance(i, bytes) else i for i in signals]  # convert bytes to str
                logger.info(f"NX Data - using auiliary_names: {signals}")
                self.generate_scannables_from_group(nx_data, dataset_names=signals)
            else:
                self.generate_scannables_from_group(nx_data)

    def generate_scannables_from_scan_fields_or_nxdata(self, hdf_file: h5py.File):
        """Generate scannables from scan_field names or default NXdata"""

        # find 'scan_fields' to generate scannables list
        if NX_SCANFIELDS in self.arrays:
            scan_fields_path = self.arrays[NX_SCANFIELDS]
            # scan_fields = hdf_file[scan_fields_path][()]
            scan_fields = names_from_scan_fields(hdf_file, scan_fields_path)
            logger.info(f"Generating Scannables from NX ScanFields: {scan_fields_path}: {scan_fields}")
            self.generate_scannables_from_names(scan_fields)
        else:
            self.generate_scannables_from_nxdata(hdf_file)

        if not self.scannables:
            logger.warning("No NXdata found, scannables not populated!")

    def generate_image_data_from_nxdetector(self):
        """find the NXdetector group and assign the image data"""
        self.image_data = {}
        if NX_DETECTOR in self.classes:
            for group_path in self.classes[NX_DETECTOR]:
                detector_name = generate_identifier(group_path)
                # detector data is stored in NXdata in dataset 'data'
                data_path = build_hdf_path(group_path, NX_DETECTOR_DATA)
                logger.debug(f"Looking for image_data at: '{data_path}'")
                if data_path in self.datasets and len(self.datasets[data_path].shape) > 1:
                    logger.info(f"Adding image_data ['{detector_name}'] = '{data_path}'")
                    self.image_data[detector_name] = data_path
                else:
                    # Use first dataset with > 2 dimensions
                    image_datasets = [
                        path for name in self.get_group_datasets(group_path)
                        if len(self.datasets[path := build_hdf_path(group_path, name)].shape) >= 3
                    ]
                    if image_datasets:
                        logger.info(f"Adding image_data ['{detector_name}'] = '{image_datasets[0]}'")
                        self.image_data[detector_name] = image_datasets[0]

        if not self.image_data:
            logger.warning("No NXdetector found, image_data not populated!")

    def populate(self, hdf_file: h5py.File, groups=None, default_entry_only=False):
        """
        Populate only datasets from default or first entry, with scannables from given groups.
        Automatically load defaults (axes, signal) and generate scannables from default group
        :param hdf_file: HDF File object
        :param groups: list of group names or NXClass names to search for datasets, within default entry
        :param default_entry_only: if True, only the first or default entry will be loaded
        """
        self.filename = hdf_file.filename

        # Add defaults to arrays
        self._default_nexus_paths(hdf_file)

        if default_entry_only:
            entries = self.classes[NX_ENTRY]  # classes[NX_ENTRY] pre-populated by _default_nexus_paths
        else:
            entries = [entry for entry in hdf_file if check_nexus_class(hdf_file.get(entry), NX_ENTRY)]

        for entry in entries:
            # find default or first entry
            nx_entry = hdf_file.get(entry)
            if nx_entry is None:
                logger.warning(
                    f"NX Entry {entry} doesn't exist - may be a missing link.\n" +
                    f"Missing link: {hdf_file.get(entry, getlink=True)}"
                )
                continue  # group may be missing due to a broken link
            hdf_path = build_hdf_path(entry)
            logger.debug(f"NX Entry: {hdf_path}")
            self.all_paths.append(hdf_path)
            self._store_group(nx_entry, hdf_path, entry)
            self._populate(nx_entry, root=hdf_path, groups=groups)  # nx_entry.name can be wrong!

        if not self.datasets:
            logger.warning("No datasets found!")

        self.generate_scannables_from_scan_fields_or_nxdata(hdf_file)

        # find the NXdetector group and assign the image data
        self.generate_image_data_from_nxdetector()

    def get_plot_data(self, hdf_file: h5py.File):
        """
        Return plotting data from scannables
        :returns: {
            'xlabel': str label of first axes
            'ylabel': str label of first signal
            'xdata': flattened array of first axes
            'ydata': flattend array of first signal
            'axes_names': list of axes names,
            'signal_names': list of signal + auxilliary signal names,
            'axes_data': list of ND arrays of data for axes,
            'signal_data': list of ND array of data for signal + auxilliary signals,
            'axes_labels': list of axes labels as 'name [units]',
            'signal_labels': list of signal labels,
            'data': dict of all scannables axes,
            'title': str title as 'filename\nNXtitle'
        if dataset is a 2D grid scan, additional rows:
            'grid_xlabel': str label of grid x-axis
            'grid_ylabel': str label of grid y-axis
            'grid_label': str label of height or colour
            'grid_xdata': 2D array of x-coordinates
            'grid_ydata': 2D array of y-coordinates
            'grid_data': 2D array of height or colour
        }
        """
        axes, signals = self.nexus_defaults()
        axes_names = [generate_identifier(path) for path in axes]
        signal_names = [generate_identifier(path) for path in signals]
        axes_units = [self.get_attr(path, NX_UNITS, '') for path in axes]
        axes_labels = [f"{name} [{unit}]" for name, unit in zip(axes_names, axes_units)]
        title = f"{self.filename}\n{self.get_data(hdf_file, NX_TITLE)}"
        data = {
            'xlabel': axes_labels[0],
            'ylabel': signal_names[0],
            'xdata': self.get_data(hdf_file, axes[0]).flatten(),
            'ydata': self.get_data(hdf_file, signals[0]).flatten(),
            'axes_names': axes_names,
            'signal_names': signal_names,
            'axes_data': [self.get_data(hdf_file, ax) for ax in axes],
            'signal_data': [self.get_data(hdf_file, sig) for sig in signals],
            'axes_labels': axes_labels,
            'signal_labels': signal_names,
            'data': self.get_scannables(hdf_file),
            'title': title
        }
        if len(axes) == 2 and len(self.scannables_shape()) == 2:
            # 2D grid scan
            data['grid_xlabel'] = axes_labels[0]
            data['grid_ylabel'] = axes_labels[1]
            data['grid_label'] = signal_names[0]
            data['grid_xdata'] = self.get_data(hdf_file, axes[0])
            data['grid_ydata'] = self.get_data(hdf_file, axes[1])
            data['grid_data'] = self.get_data(hdf_file, signals[0])
        return data