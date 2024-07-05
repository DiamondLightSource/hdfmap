from unittest import TestCase
import hdfmap

# FILE_HKL = "/scratch/grp66007/data/i16/2024/mm36462-1/1049598.nxs"  # hkl scan, pilatus
# FILE_NEW_NEXUS = "/scratch/grp66007/data/i16/example/1040323.nxs"  # new nexus format
# FILE_MSMAPPER = "/scratch/grp66007/data/i16/2024/mm36772-1/processing/1043504_rsmap.nxs"  # msmapper file
FILE_NEW_NEXUS = 'data/1040323.nxs'  # new nexus format


class TestNexusMap(TestCase):

    def setUp(self):
        self.map = hdfmap.NexusMap()
        self.map.debug(True)
        with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
            self.map.populate(hdf, groups=['instrument', 'measurement'], default_entry_only=False)

    def test_populate(self):
        self.assertEqual(len(self.map.datasets), 431, "Wrong number of datasets")
        self.assertEqual(len(self.map.combined), 472, "Wrong number of names in map.combined")
        self.assertEqual(self.map.scannables_length(), 21, "Wrong length for scannables")
        self.assertEqual(self.map['axes'], '/entry/measurement/h', "Wrong address for default axes")
        self.assertEqual(self.map.get_image_address(), '/entry/instrument/pil3_100k/data', "Wrong image address")


