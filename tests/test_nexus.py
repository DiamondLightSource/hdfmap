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


def test_default_scannables_path():
    """
    Ensure that paths in scannables always point to the default path
    """
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        hdf_map = hdfmap.NexusMap(hdf)
        hdf_map.generate_scannables_from_nxdata(hdf)  # this does not work if scan_fields is used.
        default_nxdata_path = hdf['entry'][hdf['entry'].attrs['default']].name
        nx_data_datasets = list(hdf[default_nxdata_path])

    for name, path in hdf_map.scannables.items():
        print(name, path, name in nx_data_datasets)
        if name in nx_data_datasets:
            dataset = hdf_map.datasets[path]
            group = dataset.parent
            assert group.default
            assert path.startswith(default_nxdata_path)


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


def test_generate_ids(hdf_map):
    xlabel, ylabel, expression = hdf_map.generate_ids('axes', 'signal', 'entry/measurement/sum')
    assert xlabel == 'h'
    assert ylabel == 'rc'
    assert expression == 'sum'
    expression, = hdf_map.generate_ids('signal/Transmission', modify_missing=False)
    assert expression == 'signal/Transmission'


def test_merge_default_names(hdf_map):
    assert hdf_map.merge_default_names('signal/gains_atten_Transmission') == 'rc/Transmission'

    hdf_map.add_named_expression(signal='sum')
    assert hdf_map.merge_default_names('signal/gains_atten_Transmission') == 'sum/Transmission'


def test_info_nexus(hdf_map):
    info = hdf_map.info_nexus()
    assert 'NXmx: [\'/entry\']' in info
    assert "NXdata: ['/entry/measurement', '/entry/pil3_100k', '/entry/pil3_100k_max_val', '/entry/pil3_100k_max_x', '/entry/pil3_100k_max_y', '/entry/pil3_100k_roi1.max_val', '/entry/pil3_100k_roi1.max_x', '/entry/pil3_100k_roi1.max_y', '/entry/pil3_100k_roi1.total', '/entry/pil3_100k_roi2.max_val', '/entry/pil3_100k_roi2.max_x', '/entry/pil3_100k_roi2.max_y', '/entry/pil3_100k_roi2.total', '/entry/pil3_100k_roi3.max_val', '/entry/pil3_100k_roi3.max_x', '/entry/pil3_100k_roi3.max_y', '/entry/pil3_100k_roi3.total', '/entry/pil3_100k_roi4.max_val', '/entry/pil3_100k_roi4.max_x', '/entry/pil3_100k_roi4.max_y', '/entry/pil3_100k_roi4.total', '/entry/pil3_100k_total']" in info
    assert "@default: ['/entry']" in info
    assert "@axes: /entry/measurement/h" in info
    assert "@signal: /entry/measurement/rc" in info
    assert "h: (21,)      : /entry/instrument/hkl/h " in info
    assert "pil3_100k: (21, 195, 487) : /entry/instrument/pil3_100k/data" in info


def test_nexus_default_names_i16():
    """
    In scan 1040323.nxs, the @signal is incorrectly set and does not
    match the last item in the scan_fields dataset, which defines scannables
    """
    with hdfmap.load_hdf(FILE_NEW_NEXUS) as hdf:
        hdf_map = hdfmap.NexusMap(hdf)
        scan_fields = hdf['/entry/scan_fields'].asstr()[...]
    # @signal, @axes are defined in @default NXdata
    default_axes_paths, default_signal_paths = hdf_map.nexus_default_paths()
    assert default_axes_paths[0] == '/entry/measurement/h'
    assert default_signal_paths[0] == '/entry/measurement/rc'

    # The same is returned using nexus_default_names
    default_axes, default_signal = hdf_map.nexus_default_names()
    assert len(default_axes) == len(default_axes_paths)
    assert len(default_signal) == len(default_signal_paths)
    assert default_axes == {'h': '/entry/instrument/hkl/h'}
    assert default_signal == {'rc': '/entry/measurement/rc'}
    assert all(name in hdf_map.scannables for name in list(default_axes) + list(default_signal))

    # Better defaults are available if scan_fields is used
    default_axes, default_signal = hdf_map.first_last_scannables(len(default_signal_paths))
    assert len(default_axes) == len(default_axes_paths)
    assert len(default_signal) == len(default_signal_paths)
    assert default_axes == {'h': '/entry/instrument/hkl/h'}
    assert default_signal == {'total': '/entry/instrument/pil3_100k/total'}


def test_nexus_default_names_i06(hdf_map):
    """
    In scan i06-353130.nxs, the first @axes points to a dataset
    that is not in scan_fields, which defines scannables.
    Also, the @signal points to a dataset with the wrong shape (detector data)
    """
    with hdfmap.load_hdf(FILE_3D_NEXUS) as hdf:
        hdf_map = hdfmap.NexusMap(hdf)
    default_axes_paths, default_signal_paths = hdf_map.nexus_default_paths()
    assert default_axes_paths[0] == '/entry/medipix/pol'
    assert default_signal_paths[0] == '/entry/medipix/data'

    default_axes, default_signal = hdf_map.nexus_default_names()
    assert default_axes == {'pol': '/entry/medipix/pol', 'energy': '/entry/medipix/energy', 'ds': '/entry/medipix/ds'}
    assert default_signal == {'Region1_meanvalue': '/entry/instrument/medipix/Region1_meanvalue'}
    assert all(name in hdf_map.scannables for name in list(default_axes) + list(default_signal))

    assert default_axes_paths[0] == list(default_axes.values())[0]
    assert default_signal_paths[0] != list(default_signal.values())[0]


def test_save_load(hdf_map):
    hdf_map.add_named_expression(**{"cmd": "(cmd|scan_command?('no_cmd'))"})
    save = hdf_map.generate_json_str()
    new_map = hdfmap.NexusMap(save)
    assert hdf_map.combined == new_map.combined
    assert hdf_map.scannables_length() == new_map.scannables_length()
    assert tuple(new_map.datasets['/entry/instrument/pil3_100k/data'].shape) == (21, 195, 487)
    cmd = new_map.eval(new_map.load_hdf(), 'cmd')
    assert cmd == 'scan hkl [0.97, 0.022, 0.112] [0.97, 0.022, 0.132] [0, 0, 0.001] MapperProc pil3_100k 1'
    assert new_map.datasets['/entry/instrument/monochromator/energy'].attrs == hdf_map.datasets['/entry/instrument/monochromator/energy'].attrs
    new_save = new_map.generate_json_str()
    assert save == new_save

