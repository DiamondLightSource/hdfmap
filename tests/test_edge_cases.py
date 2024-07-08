from unittest import TestCase
import json
import hdfmap


def check_nexus(*args, **kwargs):
    mymap = hdfmap.create_nexus_map(*args, **kwargs)
    print(f"\n{repr(mymap)}")
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
        image = mymap.get_image(hdf, index=0)  # index=None finds some bad files where detector only contains 1 image
        print(f"scan_command: {cmd if cmd else None}")
        print(f"        axes: {axes.shape if axes is not None else None}")
        print(f"      signal: {signal.shape if signal is not None else None}")
        print(f"       image: {image.shape if image is not None else None}")
    print(f"---")
    return mymap


# Edge case files, create this list from create_test_files.py
with open('data/test_files.json', 'r') as f:
    CHECK_FILES = json.load(f)
FOLDERS = [
    '/scratch/grp66007/data/i16/example',
]


class Test(TestCase):

    def test_edge_cases(self):
        n = 0
        for chk in CHECK_FILES:
            mymap = hdfmap.create_nexus_map(chk['filename'])
            self.assertIsInstance(mymap, hdfmap.NexusMap, f"{chk['filename']} is not NexusMap")
            self.assertEqual(len(mymap.combined), chk['len_combined'],
                             f"{chk['filename']} has wrong size of combined")
            self.assertEqual(len(mymap.scannables), chk['len_scannables'],
                             f"{chk['filename']} has wrong size of scannables")
            self.assertEqual(mymap.scannables_length(), chk['scannables_length'],
                             f"{chk['filename']} has wrong scannables_length")
            self.assertEqual(mymap['scan_command'], chk['scan_command'],
                             f"{chk['filename']} has wrong scan_command")
            self.assertEqual(mymap['axes'], chk['axes'],
                             f"{chk['filename']} has wrong axes")
            self.assertEqual(mymap['signal'], chk['signal'],
                             f"{chk['filename']} has wrong signal")
            self.assertEqual(mymap.get_image_address(), chk['image'],
                             f"{chk['filename']} has wrong image address")
            n += 1
        print(f"Completed {n} edge case files")

    def test_files_in_folders(self):
        for folder in FOLDERS:
            files = hdfmap.list_files(folder)  # returns empty if folder doesn't exist
            for file in files:
                mymap = check_nexus(file)
                self.assertIsInstance(mymap, hdfmap.NexusMap, f"{file} is not NexusMap")

