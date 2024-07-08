from unittest import TestCase
import h5py
import hdfmap


def check_nexus(*args, **kwargs):
    mymap = hdfmap.create_nexus_map(*args, **kwargs)
    print(f"\n{repr(mymap)}")
    print(f"")
    print(f"  N combined: {len(mymap.combined)}")
    print(f"N scannables: {len(mymap.scannables)}")
    print(f"      length: {mymap.scannables_length()}")
    print('Addresses:')
    print(f"scan_command: {mymap['scan_command']}")
    print(f"        axes: {mymap['axes']}")
    print(f"      signal: {mymap['signal']}")
    print(f"       image: {mymap.get_image_address()}")
    print('Data:')
    with hdfmap.load_hdf(mymap.filename) as hdf:
        cmd = mymap.get_data(hdf, 'scan_command')
        axes = mymap.get_data(hdf, 'axes')
        signal = mymap.get_data(hdf, 'signal')
        image = mymap.get_image(hdf)
        print(f"scan_command: {cmd if cmd else None}")
        print(f"        axes: {axes.shape if axes is not None else None}")
        print(f"      signal: {signal.shape if signal is not None else None}")
        print(f"       image: {image.shape if image is not None else None}")
    print(f"\n")
    return mymap


FILES = []
FOLDERS = []


class Test(TestCase):

    def setUp(self):
        self.files = [file for file in FILES if h5py.is_hdf5(file)]
        for folder in FOLDERS:
            self.files += hdfmap.list_files(folder)

    def test_edge_cases(self):
        for file in self.files:
            mymap = check_nexus(file)
            self.assertIsInstance(mymap, hdfmap.NexusMap, f"{file} is not NexusMap")