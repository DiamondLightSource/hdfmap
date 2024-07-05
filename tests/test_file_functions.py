from unittest import TestCase
import hdfmap.file_functions as ff


class Test(TestCase):
    def setUp(self):
        self.files = ff.list_files('data')

    def test_create_hdf_map(self):
        for file in self.files:
            self.assertIsInstance(ff.create_hdf_map(file), ff.HdfMap, f"{file} is not HdfMap")

    def test_create_nexus_map(self):
        for file in self.files:
            self.assertIsInstance(ff.create_nexus_map(file), ff.NexusMap, f"{file} is not NexusMap")

    def test_hdf_data(self):
        self.assertEqual(len(ff.hdf_data(self.files, 'en')), len(self.files))
        self.assertEqual(len(ff.hdf_data(self.files[0], ['en', 'sum'])), 2)

    def test_hdf_eval(self):
        file = self.files[0]
        mymap = ff.create_hdf_map(file)
        expr = "int(total[0] / Transmission)"
        with ff.load_hdf(file) as hdf:
            out = mymap.eval(hdf, expr)
        self.assertEqual(ff.hdf_eval(file, expr), out, "expression output doesn't match")

    def test_hdf_format(self):
        file = self.files[0]
        mymap = ff.create_hdf_map(file)
        expr = "energy is {en:.2f} keV"
        with ff.load_hdf(file) as hdf:
            out = mymap.format_hdf(hdf, expr)
        self.assertEqual(ff.hdf_format(file, expr), out, "expression output doesn't match")

    def test_hdf_image(self):
        file = self.files[0]
        mymap = ff.create_hdf_map(file)
        with ff.load_hdf(file) as hdf:
            image = mymap.get_image(hdf)
        self.assertEqual(ff.hdf_image(file).shape, image.shape, "image doesn't match")
