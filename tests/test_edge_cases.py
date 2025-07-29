from os import path
import json
import hdfmap
import hdfmap.hdf_loader

from . import only_dls_file_system

# Edge case files, create this list from create_test_files.py
TEST_FILES = path.join(path.dirname(__file__), 'data', 'test_files.json')
with open(TEST_FILES, 'r') as f:
    CHECK_FILES = json.load(f)


@only_dls_file_system
def test_edge_cases():
    n = 0
    for chk in CHECK_FILES:
        assert path.isfile(chk['filename']) is True, f"{chk['filename']} doesn't exist"
        mymap = hdfmap.create_nexus_map(chk['filename'])
        assert isinstance(mymap, hdfmap.NexusMap), f"{chk['filename']} is not NexusMap"
        assert len(mymap.combined) == chk['len_combined'], "{chk['filename']} has wrong size of combined"
        assert len(mymap.scannables) == chk['len_scannables'], f"{chk['filename']} has wrong size of scannables"
        assert mymap.scannables_length() == chk['scannables_length'], f"{chk['filename']} has wrong scannables_length"
        assert mymap.get_path('scan_command') == chk['scan_command'], f"{chk['filename']} has wrong scan_command"
        assert mymap.get_path('axes') == chk['axes'], f"{chk['filename']} has wrong axes"
        assert mymap.get_path('signal') == chk['signal'], f"{chk['filename']} has wrong signal"
        assert mymap.get_image_path() == chk['image'], f"{chk['filename']} has wrong image path"
        assert mymap.info_nexus(scannables=True, image_data=True, metadata=True) == chk['string']
        n += 1
    print(f"Completed {n} edge case files")


@only_dls_file_system
def test_old_i16_file():
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1040311.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        value, address = mymap.eval(hdf, 'sum.sum(), _sum')
    assert abs(value + 407) < 0.01, 'expression "sum.sum()" gives wrong result'
    assert address == '/entry1/measurement/sum', 'expression "_sum" returns wrong address'


@only_dls_file_system
def test_new_i16_file():
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1040323.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        h, k, l, hkl, _h, fname = mymap.eval(hdf, 'h, k, l, hkl, _h, filename')
    assert h.shape == (21,), 'expression "h" has wrong shape'
    assert hkl == '--', 'default for expression "hkl" is incorrect'
    assert fname == '1040323.nxs'


@only_dls_file_system
def test_msmapper_file():
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/processed/1098101_msmapper.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    assert mymap['unit_cell'] == '/entry0/sample/unit_cell', 'link to old file incorrect'
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        a, b, c, alpha, beta, gamma = mymap.eval(hdf, 'unit_cell')
    assert gamma > 1.0, 'unit cell incorrect'


