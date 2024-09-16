import pytest
import os
import hdfmap

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_HKL = DATA_FOLDER + "/1049598.nxs"  # hkl scan, pilatus


@pytest.fixture
def hdf_map():
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        hdf_map = hdfmap.HdfMap(hdf)
    yield hdf_map


def test_populate(hdf_map):
    assert len(hdf_map.datasets) == 360, "Wrong number of datasets loaded"
    assert len(hdf_map.combined) == 570, "Wrong number of names in map.combined"


def test_most_common_size(hdf_map):
    assert hdf_map.most_common_size() == 101, "Most common size should be 101"


def test_scannables_length(hdf_map):
    assert hdf_map.scannables_length() == 101, "scannables length should be 101"


def test_generate_scannables(hdf_map):
    hdf_map.generate_scannables(3)
    assert hdf_map.scannables_length() == 3, "Scannable size should be 3"


def test_get_item(hdf_map):
    assert hdf_map['sum'] == '/entry1/pil3_100k/sum', '__get_item__ failed'
    assert 'sum' in hdf_map, '__contains__ failed'
    assert len([path for path in hdf_map]) == 570, '__iter__ failed'


def test_get_path(hdf_map):
    assert hdf_map.get_path('/entry1/measurement/sum') == '/entry1/measurement/sum', 'path is wrong'
    assert hdf_map.get_path('sum') == '/entry1/pil3_100k/sum', 'name is wrong'
    assert hdf_map.get_path('NXdata') == '/entry1/measurement', 'class is wrong'


def test_get_group_path(hdf_map):
    assert hdf_map.get_group_path('sum') == '/entry1/pil3_100k'


def test_find(hdf_map):
    assert len(hdf_map.find_paths('eta')) == 17, "Can't find eta in names"
    assert len(hdf_map.find_paths('eta', False)) == 17, "Can't find eta anywhere"


def test_find_attr(hdf_map):
    assert len(hdf_map.find_attr('signal')) == 4, "Wrong number of 'signal' attributes found"


def test_get_image_path(hdf_map):
    assert hdf_map.get_image_path() == '/entry1/pil3_100k/data'


def test_get_group_datasets(hdf_map):
    assert len(hdf_map.get_group_datasets('NXdata')) == 29


"--------------------------------------------------------"
"---------------------- FILE READERS --------------------"
"--------------------------------------------------------"


def test_get_data(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        en = hdf['/entry1/before_scan/mono/en'][()]
        h = hdf['/entry1/measurement/h'][()]
        cmd = hdf['/entry1/scan_command'].asstr()[()]
        assert hdf_map.get_data(hdf, 'en') == en, "'en' produces wrong result"
        assert (hdf_map.get_data(hdf, 'h') == h).all(), "'h' produces wrong result"
        assert hdf_map.get_data(hdf, 'scan_command')[:8] == cmd[:8], "'cmd' produces wrong result"


def test_get_string(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        assert hdf_map.get_string(hdf, 'en') == '3.5800002233729673'
        assert hdf_map.get_string(hdf, 'h') == 'float64 (101,)'
        assert hdf_map.get_string(hdf, 'start_time') == "'2024-05-17 14:13:27.025000'"


def test_get_image(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        assert hdf_map.get_image(hdf).shape == (195, 487)


def test_get_dataholder(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        d = hdf_map.get_dataholder(hdf)
    assert d.metadata.filepath == FILE_HKL, "Filename not included in data object metadata"
    assert int(100 * d.metadata.en) == 358, "metadata energy is wrong"
    assert d.h.shape == (101,), "scannable h is wrong shape"


def test_get_metadata(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        meta = hdf_map.get_metadata(hdf)
        meta_small = hdf_map.get_metadata(hdf, name_list=['scan_command', 'incident_energy'])
        meta_string = hdf_map.get_metadata(hdf, string_output=True)
    assert len(meta) == 423, "Length of metadata wrong"
    assert meta['filename'] == '1049598.nxs', "filename is wrong"
    assert abs(meta_small['incident_energy'] - 3.58) < 0.01, "Energy is wrong"
    cmd = "'scan hkl [-0.05, -7.878e-16, 0.933] [0.05, -7.878e-16, 0.933] [0.001, 0, 0] BeamOK pil3_100k 1 roi2 roi1'"
    assert meta_string['scan_command'] == cmd
    assert meta_string['ppchi'] == '-44.999994057'


def test_create_metadata_list(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        meta = hdf_map.create_metadata_list(hdf)
    assert len(meta) == 11391, "Length of metadata list wrong"


def test_get_scannables(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.get_scannables(hdf)
    assert len(scannables) == 131, "Length of scannables is wrong"


def test_get_scannables_array(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.get_scannables_array(hdf)
        assert scannables.shape == (129, 101), "scannables array is wrong shape"


def test_create_scannables_table(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.create_scannables_table(hdf, '\t')
        assert len(scannables) == 165703, "scannables str is wrong length"


def test_eval(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.eval(hdf, 'int(np.max(sum / Transmission / count_time))')
        assert out == 6533183, "Expression output gives wrong result"


def test_format_hdf(hdf_map):
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.format_hdf(hdf, 'The energy is {en:.3} keV')
        assert out == 'The energy is 3.58 keV', "Expression output gives wrong result"
