import pytest
import os
import hdfmap

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_NEW_NEXUS = DATA_FOLDER + '/1040323.nxs'  # new nexus format
FILE_3D_NEXUS = DATA_FOLDER + '/i06-353130.nxs'  # new nexus format

hdfmap.set_all_logging_level('debug')


@pytest.fixture
def hdf_map():
    hdf_map = hdfmap.NexusMap()
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        hdf_map.populate(hdf, groups=['instrument', 'measurement'], default_entry_only=False)
    yield hdf_map


def test_populate(hdf_map):
    assert len(hdf_map.datasets) == 431, "Wrong number of datasets"
    assert len(hdf_map.combined) == 973, "Wrong number of names in map.combined"
    assert hdf_map.scannables_length() == 21, "Wrong length for scannables"
    assert hdf_map['axes'] == '/entry/measurement/h', "Wrong path for default axes"
    assert hdf_map.get_image_path() == '/entry/instrument/pil3_100k/data', "Wrong image path"
    assert hdf_map['IMAGE'] == '/entry/instrument/pil3_100k/data', "Wrong image path"


def test_dataset_names(hdf_map):
    assert hdf_map['s5xgap'] == '/entry/instrument/s5/x_gap', "LocalName: 's5xgap' points to wrong path"
    assert hdf_map['s5_x_gap'] == '/entry/instrument/s5/x_gap', "GroupName: 's5_x_gap' points to wrong path"
    assert hdf_map['x_gap'] == '/entry/instrument/s7/x_gap', "Name: 'x_gap' points to wrong path"


def test_find_datasets(hdf_map):
    assert len(hdf_map.find_datasets('NXslit', 'x_gap')) == 7
    assert len(hdf_map.find_datasets('NXdetector', 'data')) == 1


def test_nexus_decimals(hdf_map):
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        out = hdf_map.get_string(hdf, 'ppth2')
        assert out == '-0.00047'
        out = hdf_map.get_string(hdf, 'pppiezo2')
        assert out == '12345.01236'

def test_nexus_eval(hdf_map):
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        out = hdf_map.eval(hdf, 'int(max(total / Transmission / count_time))')
        assert out == 70, "Expression output gives wrong result"
        path = hdf_map.eval(hdf, '_axes')
        assert path == '/entry/measurement/h', "Wrong axes path"
        out = hdf_map.eval(hdf, '__axes')
        assert out == 'h', "Wrong axes name"
        out = hdf_map.eval(hdf, 's_ppy')  # example uses decimals and units
        assert out == '-7.4871 mm', "Incorrect label"
        out = hdf_map.eval(hdf, 'idgap@units')
        assert out == 'mm', "Incorrect attribute"
        out = hdf_map.eval(hdf, '(cmd|nout|scan_command)')
        assert out == 'scan hkl [0.97, 0.022, 0.112] [0.97, 0.022, 0.132] [0, 0, 0.001] MapperProc pil3_100k 1'
        out = hdf_map.eval(hdf, '(gains_atten|atten?(0))')
        assert out == 0, "default expression failed"
        out = hdf_map.eval(hdf, '"pol in" if abs(delta_offset) < 0.1 and abs(thp) > 20 else "pol out"')
        assert out == 'pol out', "expression failed"
        title = hdf_map.format_hdf(hdf, '{filename}: {scan_command}')
        correct = '1040323.nxs: scan hkl [0.97, 0.022, 0.112] [0.97, 0.022, 0.132] [0, 0, 0.001] MapperProc pil3_100k 1'
        assert title == correct, "Expression output gives wrong result"
        out = hdf_map.format_hdf(hdf, '({mean(h):.3g},{mean(k):.3g},{mean(l):.3g})')
        assert out == '(0.97,0.0221,0.122)', "Expression output gives wrong result"


def test_plot_data(hdf_map):
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        data = hdf_map.get_plot_data(hdf)
        assert 'title' in data, 'plot_data missing attributes'
        assert data['ydata'].shape == (21, ), "plot_data['ydata'] is the wrong shape"


def test_3d_scan():
    hdf_map = hdfmap.create_nexus_map(FILE_3D_NEXUS)
    assert hdf_map.scannables_length() == 80, "Scannables have the wrong length"
    axes, signals = hdf_map.nexus_default_paths()
    assert len(axes) == 3, "Number of default axes is wrong"
    assert signals[0] == '/entry/medipix/data', "Incorrect default signal"
    with hdf_map.load_hdf() as hdf:
        table = hdf_map.create_scannables_table(hdf)
        assert table.count('\n') == 80, "table has the wrong length"
        assert len(table) == 4085, "wrong number of characters in table"

        array = hdf_map.get_scannables_array(hdf)
        assert array.shape == (4, 80)
        assert array[0].shape == (80, )

        structured_array = hdf_map.get_scannables_array(hdf, return_structured_array=True)
        assert structured_array['energy'].shape == (80, )

        image = hdf_map.get_image(hdf, index=None)
        assert image.shape == (512, 512), "image shape is wrong"


