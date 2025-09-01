import h5py
import pytest
import sys
import os
import datetime
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
    assert len(hdf_map.combined) == 895, "Wrong number of names in map.combined"


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
    assert len([path for path in hdf_map]) == 895, '__iter__ failed'


def test_get_path(hdf_map):
    assert hdf_map.get_path('/entry1/measurement/sum') == '/entry1/measurement/sum', 'path is wrong'
    assert hdf_map.get_path('sum') == '/entry1/pil3_100k/sum', 'name is wrong'
    assert hdf_map.get_path('NXdata') == '/entry1/measurement', 'class is wrong'


def test_get_group_path(hdf_map):
    assert hdf_map.get_group_path('sum') == '/entry1/pil3_100k'


def test_find(hdf_map):
    assert len(hdf_map.find_paths('eta')) == 11, "Can't find eta in names"
    assert len(hdf_map.find_paths('eta', False)) == 11, "Can't find eta anywhere"


def test_find_attr(hdf_map):
    assert len(hdf_map.find_attr('signal')) == 4, "Wrong number of 'signal' attributes found"


def test_get_image_path(hdf_map):
    assert hdf_map.get_image_path() == '/entry1/pil3_100k/data'


def test_get_group_datasets(hdf_map):
    assert len(hdf_map.get_group_datasets('NXdata')) == 29


def test_find_groups(hdf_map):
    assert len(hdf_map.find_groups('NXentry', 'measurement')) == 1


def test_find_datasets(hdf_map):
    assert len(hdf_map.find_datasets('measurement', 'sum')) == 1


"--------------------------------------------------------"
"---------------------- FILE READERS --------------------"
"--------------------------------------------------------"


def test_get_data(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        en = hdf['/entry1/before_scan/mono/en'][()]
        h = hdf['/entry1/measurement/h'][()]
        cmd = hdf['/entry1/scan_command'].asstr()[()]
        scanno = int(hdf['/entry1/entry_identifier'][()])
        assert hdf_map.get_data(hdf, 'en') == en, "'en' produces wrong result"
        assert (hdf_map.get_data(hdf, 'h') == h).all(), "'h' produces wrong result"
        assert hdf_map.get_data(hdf, 'scan_command')[:8] == cmd[:8], "'cmd' produces wrong result"
        assert hdf_map.get_data(hdf, 'entry_identifier') == scanno, "'entry_identifier' gives wrong result"


def test_get_date(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        time = hdf_map.get_data(hdf, 'start_time')
        if sys.version_info >= (3, 11, 0):
            # datetime.fromisoformat only accepts Nexus timestamps from Python 3.11
            assert isinstance(time, datetime.datetime)
            assert time.strftime('%y/%m/%d %H:%M') == '24/05/17 15:13', "'start_time' gives wrong time"
        else:
            assert isinstance(time, str), "'start_time' is wrong type"


def test_get_string(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        assert hdf_map.get_string(hdf, 'en') == '3.5800002233729673'
        assert hdf_map.get_string(hdf, 'h') == 'float64 (101,)'
        if sys.version_info >= (3, 11, 0):
            assert hdf_map.get_string(hdf, 'start_time') == "'2024-05-17 15:13:27.025000+01:00'"
        else:  # without intermediate conversion to datetime
            assert hdf_map.get_string(hdf, 'start_time') == "'2024-05-17T15:13:27.025+01'"


def test_get_image(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        assert hdf_map.get_image(hdf).shape == (195, 487)


def test_get_dataholder(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        d = hdf_map.get_dataholder(hdf)
    assert d.metadata.filepath == FILE_HKL, "Filename not included in data object metadata"
    assert int(100 * d.metadata.en) == 358, "metadata energy is wrong"
    assert d.h.shape == (101,), "scannable h is wrong shape"


def test_get_metadata(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        meta = hdf_map.get_metadata(hdf)
        meta_small = hdf_map.get_metadata(hdf, name_list=['scan_command', 'incident_energy'])
        meta_string = hdf_map.get_metadata(hdf, string_output=True)
    assert len(meta) == 221, "Length of metadata wrong"
    assert meta['filename'] == '1049598.nxs', "filename is wrong"
    assert abs(meta_small['incident_energy'] - 3.58) < 0.01, "Energy is wrong"
    cmd = "'scan hkl [-0.05, -7.878e-16, 0.933] [0.05, -7.878e-16, 0.933] [0.001, 0, 0] BeamOK pil3_100k 1 roi2 roi1'"
    assert meta_string['scan_command'] == cmd
    assert meta_string['ppchi'] == '-44.999994057'


def test_create_metadata_list(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        meta = hdf_map.create_metadata_list(hdf)
    if sys.version_info >= (3, 11, 0):
        assert len(meta) == 5280, "Length of metadata list wrong"
    else:  # length of time strings changes
        assert len(meta) == 5312, "Length of metadata list wrong"


def test_get_scannables(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.get_scannables(hdf)
    assert len(scannables) == 37, "Length of scannables is wrong"


def test_get_scannables_array(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.get_scannables_array(hdf)
        assert scannables.shape == (36, 101), "scannables array is wrong shape"


def test_create_scannables_table(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        scannables = hdf_map.create_scannables_table(hdf, '\t')
        assert len(scannables) == 45365, "scannables str is wrong length"


def test_eval(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.eval(hdf, 'int(max(sum / Transmission / count_time))')
        assert out == 6533183, "Expression output gives wrong result"


def test_eval_patterns(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        filename, filepath = hdf_map.eval(hdf, 'filename, filepath')
        assert filename == '1049598.nxs', "filename is incorrect"
        assert filepath.endswith(filename), "filepath is incorrect"
        path, name = hdf_map.eval(hdf, '_scan_command, __measurement_h')
        assert path == '/entry1/scan_command', '_*name* not working'
        assert name == 'h', '__*name* not working'
        string = hdf_map.eval(hdf, 's_fast_pixel_direction')
        assert string == '0.172 mm', 's_*name* not working'
        dataset_energy, dataset_image = hdf_map.eval(hdf, 'd_incident_energy, d_IMAGE')
        assert dataset_energy, 'incident_energy dataset is closed'
        assert dataset_image.shape == (101, 195, 487), "image dataset is wrong"
        attribute = hdf_map.eval(hdf, 'fast_pixel_direction@units')
        assert attribute == 'mm', "incorrect units for fast_pixel_direction dataset"
        correct, default = hdf_map.eval(hdf, 'source_probe?("none"), source_cat?("meow")')
        assert correct == 'x-ray', 'source_probe not read correctly'
        assert default == 'meow', 'default value not returned'
        # alternate names
        out = hdf_map.eval(hdf, '(analyser_name|pa_detector_name)')
        assert out == 'Au111', 'incorrect alternate found'
        out = hdf_map.eval(hdf, '(analyzer_name|pa_detector_name)')
        assert out == 'Merlin', 'incorrect alternate found'
        out = hdf_map.eval(hdf, '(one|two|three?(4))')
        assert out == 4, 'should return default value'
    # check closed state of datasets
    assert not dataset_energy, f"{dataset_energy} is still open"
    # dataset_image points to an external file which is not closed,
    # this is a known bug and may be fixed in the future (current date July 2025)
    # See: https://github.com/h5py/h5py/issues/2454
    assert dataset_image, f"{dataset_image} has been closed"


def test_call(hdf_map):
    value, address, ub_matrix = hdf_map('sum.sum(), _sum, isnan(ub_matrix)')
    assert abs(value - 38127391) < 0.01, 'expression "sum.sum()" gives wrong result'
    assert address == '/entry1/pil3_100k/sum', 'expression "_sum" returns wrong address'
    assert ub_matrix.shape == (3, 3), '"ub_matrix" is missing or has wrong shape'


def test_format_hdf(hdf_map):
    with hdfmap.hdf_loader.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.format_hdf(hdf, 'The energy is {en:.3} keV')
        assert out == 'The energy is 3.58 keV', "Expression output gives wrong result"


def test_eval_local_data(hdf_map):
    hdf_map.add_local(new_var='testing-testing', Transmission=10.)
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.eval(hdf, 'new_var')
        assert out == 'testing-testing', "Expression output gives wrong result"
        out = hdf_map.eval(hdf, 'int(max(sum / Transmission / count_time))')
        assert out == 653318, "Expression output gives wrong result"


def test_eval_named_expression(hdf_map):
    hdf_map.add_named_expression(
        norm_data='int(max(sum / Transmission / count_time))',
        my_path=hdf_map['incident_energy']
    )
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.eval(hdf, 'norm_data')
        assert out == 6533183, "Expression output gives wrong result"
        out = hdf_map.eval(hdf, 'my_path')
        assert abs(out - 3.58) < 0.001, "Expression output gives wrong result"


def test_roi(hdf_map):
    hdf_map.add_roi('nroi1', 'pil3_centre_j', 'pil3_centre_i', 101, 21)
    with hdfmap.load_hdf(FILE_HKL) as hdf:
        out = hdf_map.eval(hdf, 'nroi1')
        assert out.shape == (101, 100, 20), "ROI output is wrong shape"
        out = hdf_map.eval(hdf, 'max(nroi1_total)')
        assert out == 199042, "ROI total gives wrong result"
        out = hdf_map.eval(hdf, 'max(nroi1_max)')
        assert out == 13191, "ROI max gives wrong result"
        out = hdf_map.eval(hdf, 'max(nroi1_min)')
        assert out == 0, "ROI min gives wrong result"
        out = hdf_map.eval(hdf, 'max(nroi1_mean)')
        assert abs(out - 99.521) < 0.001, "ROI mean gives wrong result"
        out = hdf_map.eval(hdf, 'max(nroi1_bkg)')
        assert out == 333259, "ROI bkg gives wrong result"
        out = hdf_map.eval(hdf, 'max(nroi1_rmbkg)')
        assert abs(out - 195839) < 0.001, "ROI rmbkg gives wrong result"


def test_interpreter(hdf_map):
    ii = hdf_map.create_interpreter()
    assert abs(ii('abs(mean(max(roi2_sum)))') - 359573) < 0.001, "Expression output gives wrong result"
    assert 'hkl' in ii('scan_command'), 'Expression output gives wrong result'
