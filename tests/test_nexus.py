import pytest
import os
import hdfmap

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_NEW_NEXUS = DATA_FOLDER + '/1040323.nxs'  # new nexus format

hdfmap.set_all_logging_level('debug')



@pytest.fixture
def hdf_map():
    hdf_map = hdfmap.NexusMap()
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        hdf_map.populate(hdf, groups=['instrument', 'measurement'], default_entry_only=False)
    yield hdf_map


def test_populate(hdf_map):
    assert len(hdf_map.datasets) == 431, "Wrong number of datasets"
    assert len(hdf_map.combined) == 634, "Wrong number of names in map.combined"
    assert hdf_map.scannables_length() == 21, "Wrong length for scannables"
    assert hdf_map['axes'] == '/entry/measurement/h', "Wrong path for default axes"
    assert hdf_map.get_image_path() == '/entry/instrument/pil3_100k/data', "Wrong image path"


def test_dataset_names(hdf_map):
    assert hdf_map['s5xgap'] == '/entry/instrument/s5/x_gap', "LocalName: 's5xgap' points to wrong path"
    assert hdf_map['s5_x_gap'] == '/entry/instrument/s5/x_gap', "GroupName: 's5_x_gap' points to wrong path"
    assert hdf_map['x_gap'] == '/entry/instrument/s7/x_gap', "Name: 'x_gap' points to wrong path"


def test_nexus_eval(hdf_map):
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        out = hdf_map.eval(hdf, 'int(np.max(total / Transmission / count_time))')
        assert out == 70, "Expression output gives wrong result"
        path = hdf_map.eval(hdf, '_axes')
        assert path == '/entry/measurement/h', "Wrong axes path"
        title = hdf_map.format_hdf(hdf, '{filename}: {scan_command}')
        correct = '1040323.nxs: scan hkl [0.97, 0.022, 0.112] [0.97, 0.022, 0.132] [0, 0, 0.001] MapperProc pil3_100k 1'
        assert title == correct, "Expression output gives wrong result"