"""
Nexus Related functions and nexus class
"""

import h5py

from .hdfmap_class import HdfMap, build_address


NX_LOCALNAME = 'local_name'
NX_DEFAULT = 'default'
NX_MEASUREMENT = 'measurement'
NX_SCANFIELDS = 'scan_fields'
NX_SIGNAL = 'signal'
NX_AXES = 'axes'
NX_DETECTOR = 'NXdetector'
NX_DETECTOR_DATA = 'data'


def default_nxentry(hdf_file: h5py.File) -> str | bytes:
    """Return the default nxentry address"""
    return hdf_file.attrs[NX_DEFAULT] if NX_DEFAULT in hdf_file.attrs else next(iter(hdf_file))


def get_nexus_axes_datasets(hdf_file: h5py.File) -> tuple[list[str], str]:
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry group in top File group
     - find "default" data group in entry
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate addresses of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_file: open HDF file object, i.e. h5py.File(...)
    :return axes_addresses: list of str hdf addresses for axes datasets
    :return signal_address: str hdf address for signal dataset
    """
    # From: https://manual.nexusformat.org/examples/python/plotting/index.html
    # find the default NXentry group
    nx_entry_name = hdf_file.attrs[NX_DEFAULT] if NX_DEFAULT in hdf_file.attrs else next(iter(hdf_file.keys()))
    nx_entry = hdf_file[nx_entry_name]
    # find the default NXdata group
    nx_data_name = nx_entry.attrs[NX_DEFAULT] if NX_DEFAULT in nx_entry.attrs else NX_MEASUREMENT
    nx_data = nx_entry[nx_data_name]
    # find the axes field(s)
    if isinstance(nx_data.attrs[NX_AXES], (str, bytes)):
        axes_addresses = [build_address(nx_entry_name, nx_data_name, nx_data.attrs[NX_AXES])]
    else:
        axes_addresses = [build_address(nx_entry_name, nx_data_name, _axes) for _axes in nx_data.attrs[NX_AXES]]
    # get the signal field
    signal_address = build_address(nx_entry_name, nx_data_name, nx_data.attrs[NX_SIGNAL])
    return axes_addresses, signal_address


def get_strict_nexus_axes_datasets(hdf_file: h5py.File) -> tuple[list[h5py.Dataset], h5py.Dataset]:
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry group in top File group
     - find "default" data group in entry
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate addresses of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_file: open HDF file object, i.e. h5py.File(...)
    :return axes_datasets: list of dataset objects for axes
    :return signal_dataset: dataset object for plot axis
    """
    # From: https://manual.nexusformat.org/examples/python/plotting/index.html
    # find the default NXentry group
    nx_entry = hdf_file[hdf_file.attrs["default"]]
    # find the default NXdata group
    nx_data = nx_entry[nx_entry.attrs["default"]]
    # find the axes field(s)
    if isinstance(nx_data.attrs["axes"], (str, bytes)):
        axes_datasets = [nx_data[nx_data.attrs["axes"]]]
    else:
        axes_datasets = [nx_data[_axes] for _axes in nx_data.attrs["axes"]]
    # find the signal field
    signal_dataset = nx_data[nx_data.attrs["signal"]]
    return axes_datasets, signal_dataset


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
    nxmap['axes'] -> return address of default axes dataset
    nxmap['signal'] -> return address of default signal dataset
    """

    def _load_defaults(self, hdf_file):
        """Load Nexus default axes and signal"""
        super()._load_defaults(hdf_file)
        try:
            axes_addresses, signal_address = get_nexus_axes_datasets(hdf_file)
            if axes_addresses and axes_addresses[0] in hdf_file:
                self.arrays[NX_AXES] = axes_addresses[0]
                if self._debug:
                    self._debug_logger(f"DEFAULT axes: {axes_addresses}")
            if signal_address in hdf_file:
                self.arrays[NX_SIGNAL] = signal_address
                if self._debug:
                    self._debug_logger(f"DEFAULT signal: {signal_address}")
        except KeyError:
            pass

    def _scannables_from_scan_fields_or_nxdata(self, hdf_file: h5py.File):
        """Generate scannables from scan_field names or default NXdata"""
        # find 'scan_fields' to generate scannables list
        if NX_SCANFIELDS in self.arrays:
            scan_fields_address = self.arrays[NX_SCANFIELDS]
            scan_fields = hdf_file[scan_fields_address][()]
            if self._debug:
                self._debug_logger(f"NX ScanFields: {scan_fields_address}: {scan_fields}")
            self.generate_scannables_from_names(scan_fields)
        else:
            # find the default NXdata group and generate the scannables list
            nx_entry = hdf_file.get(default_nxentry(hdf_file))
            nx_data = nx_entry.get(nx_entry.attrs[NX_DEFAULT] if NX_DEFAULT in nx_entry.attrs else NX_MEASUREMENT)
            if nx_data:
                if self._debug:
                    self._debug_logger(f"NX Data: {nx_data.name}")
                self.generate_scannables_from_group(nx_data)

        if self._debug and not self.scannables:
            self._debug_logger("!!!Warning: No NXdata found, scannables not populated!")

    def _image_data_from_nxdetector(self):
        """find the NXdetector group and assign the image data"""
        self.image_data = {}
        if NX_DETECTOR in self.classes:
            for group_address in self.classes[NX_DETECTOR]:
                detector_name = group_address.split('/')[-1]
                data_address = build_address(group_address, NX_DETECTOR_DATA)
                if data_address in self.datasets and len(self.get_shape(data_address)) > 1:
                    self.image_data[detector_name] = data_address

        if self._debug and not self.image_data:
            self._debug_logger("!!!Warning: No NXdetector found, image_data not populated!")

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
        self._load_defaults(hdf_file)

        if default_entry_only:
            entries = [default_nxentry(hdf_file)]
        else:
            entries = hdf_file.keys()

        for entry in entries:
            # find default or first entry
            nx_entry = hdf_file.get(entry)
            if nx_entry is None:
                if self._debug:
                    self._debug_logger(f"NX Entry {entry} doesn't exist - may be a missing link.")
                    self._debug_logger(f"Missing link: {hdf_file.get(entry, getlink=True)}")
                continue  # group may be missing due to a broken link
            address = build_address(entry)
            if self._debug:
                self._debug_logger(f"NX Entry: {address}")
            self._store_group(nx_entry, address, entry)
            self._populate(nx_entry, top_address=address, groups=groups)  # nx_entry.name can be wrong!

            # find 'scan_fields' to generate scannables list
            if NX_SCANFIELDS in self.arrays:
                scan_fields_address = self.arrays[NX_SCANFIELDS]
                scan_fields = hdf_file[scan_fields_address][()]
                if self._debug:
                    self._debug_logger(f"NX ScanFields: {scan_fields_address}: {scan_fields}")
                self.generate_scannables_from_names(scan_fields)
            else:
                # find the default NXdata group and generate the scannables list
                nx_data = nx_entry.get(nx_entry.attrs[NX_DEFAULT] if NX_DEFAULT in nx_entry.attrs else NX_MEASUREMENT)
                if nx_data:
                    if self._debug:
                        self._debug_logger(f"NX Data: {nx_data.name}")
                    self.generate_scannables_from_group(nx_data)

        if self._debug and not self.scannables:
            self._debug_logger("!!!Warning: No NXdata found, scannables not populated!")
        if self._debug and not self.datasets:
            self._debug_logger("!!!Warning: No datasets found!")

        # TODO: replace per-entry generation of scannables with single generation from the default nxdata or scan_fields
        # self._scannables_from_scan_fields_or_nxdata(hdf_file)

        # find the NXdetector group and assign the image data
        # TODO: clean this up so so image data only gives {'detector_name': address}
        # self._image_data_from_nxdetector()
        if NX_DETECTOR in self.classes:
            for class_address in self.classes[NX_DETECTOR]:
                dataset_address = build_address(class_address, NX_DETECTOR_DATA)
                if self._debug:
                    self._debug_logger(f"NX Detector: {dataset_address} : {hdf_file.get(dataset_address)}")
                if dataset_address in hdf_file and hdf_file[dataset_address].ndim > 1:
                    if NX_DETECTOR not in self.image_data:
                        self.image_data[NX_DETECTOR] = dataset_address  # first NXdetector
                    self.image_data['_'.join(dataset_address.split('/')[-2:])] = dataset_address  # e.g. pil3_100k_data

    def get_image_address(self) -> str:
        """Return address of first dataset named 'data'"""
        if self._default_image_address:
            return self._default_image_address
        # TODO: replace this with fist image_data field
        if NX_DETECTOR in self.image_data:
            return self.image_data[NX_DETECTOR]
