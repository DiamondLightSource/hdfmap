from unittest import TestCase
import hdfmap


# FILE_HKL = "/scratch/grp66007/data/i16/2024/mm36462-1/1049598.nxs"  # hkl scan, pilatus
FILE_HKL = "data/1049598.nxs"  # hkl scan, pilatus


class TestHdfMap(TestCase):

    def setUp(self):
        self.map = hdfmap.HdfMap()
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            self.map.populate(hdf)
        self.map.generate_scannables(self.map.most_common_size())

    def test_populate(self):
        self.assertEqual(len(self.map.datasets), 360, "Wrong number of datasets loaded")
        self.assertEqual(len(self.map.combined), 283, "Wrong number of names in map.combined")

    def test_most_common_size(self):
        self.assertEqual(self.map.most_common_size(), 101, "Most common size should be 101")

    def test_scannables_length(self):
        self.assertEqual(self.map.scannables_length(), 101, "scannables length should be 101")

    def test_generate_scannables(self):
        self.map.generate_scannables(3)
        self.assertEqual(self.map.scannables_length(), 3, "Scannable size should be 3")

    def test_get_address(self):
        self.assertEqual(self.map.get_address('/entry1/measurement/sum'), '/entry1/measurement/sum',
                         'address is wrong')
        self.assertEqual(self.map.get_address('sum'), '/entry1/pil3_100k/sum',
                         'name is wrong')
        self.assertEqual(self.map.get_address('NXdata'), '/entry1/measurement',
                         'class is wrong')

    def test_get_group_address(self):
        self.assertEqual(self.map.get_group_address('sum'), '/entry1/pil3_100k')

    def test_find(self):
        self.assertEqual(len(self.map.find('eta')), 10, "Can't find eta in names")
        self.assertEqual(len(self.map.find('eta', False)), 11, "Can't find eta anywhere")

    def test_find_attr(self):
        self.assertEqual(len(self.map.find_attr('signal')), 4,
                         "Wrong number of 'signal' attributes found")

    def test_get_image_address(self):
        self.assertEqual(self.map.get_image_address(), '/entry1/pil3_100k/data')

    def test_get_group_datasets(self):
        self.assertEqual(len(self.map.get_group_datasets('NXdata')), 29)

    "--------------------------------------------------------"
    "---------------------- FILE READERS --------------------"
    "--------------------------------------------------------"

    def test_get_data(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            en = hdf['/entry1/before_scan/mono/en'][()]
            h = hdf['/entry1/measurement/h'][()]
            cmd = hdf['/entry1/scan_command'][()]
            self.assertEqual(self.map.get_data(hdf, 'en'), en,
                             "'en' produces wrong result")
            self.assertTrue((self.map.get_data(hdf, 'h') == h).all(),
                             "'h' produces wrong result")
            self.assertEqual(self.map.get_data(hdf, 'scan_command'), cmd,
                             "'cmd' produces wrong result")

    def test_get_image(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            self.assertEqual(self.map.get_image(hdf, None).shape, (195, 487))

    def test_get_data_object(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            d = self.map.get_data_block(hdf)
            self.assertEqual(d.metadata.filepath, FILE_HKL, "Filename not included in data object metadata")
            self.assertEqual(int(100*d.metadata.en), 358, "metadata energy is wrong")
            self.assertEqual(d.h.shape, (101, ), "scannable h is wrong shape")

    def test_get_metadata(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            meta = self.map.get_metadata(hdf)
            self.assertEqual(len(meta), 213, "Length of metadata wrong")
            self.assertEqual(meta['filename'], '1049598.nxs', "filename is wrong")

    def test_get_scannables(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            scannables = self.map.get_scannables(hdf)
            self.assertEqual(len(scannables), 66)

    def test_get_scannables_array(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            scannables = self.map.get_scannables_array(hdf)
            self.assertEqual(scannables.shape, (65, 101), "scannables array is wrong shape")

    def test_get_scannables_str(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            scannables = self.map.get_scannables_str(hdf, '\t')
            self.assertEqual(len(scannables), 82500, "scannables str is wrong length")

    def test_eval(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            out = self.map.eval(hdf, 'int(np.max(sum / Transmission / count_time))')
            self.assertEqual(out, 6533183, "Expression output gives wrong result")

    def test_format_hdf(self):
        with hdfmap.load_hdf(FILE_HKL) as hdf:
            out = self.map.format_hdf(hdf, 'The energy is {en:.3} keV')
            self.assertEqual(out, 'The energy is 3.58 keV', "Expression output gives wrong result")
