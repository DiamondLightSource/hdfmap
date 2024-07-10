from unittest import TestCase
import os
import json
import hdfmap


# Edge case files, create this list from create_test_files.py
with open('data/test_files.json', 'r') as f:
    CHECK_FILES = json.load(f)


class Test(TestCase):

    def test_edge_cases(self):
        n = 0
        for chk in CHECK_FILES:
            self.assertEqual(os.path.isfile(chk['filename']), True, f"{chk['filename']} doesn't exist")
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
