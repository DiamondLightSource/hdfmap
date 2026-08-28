from os import path
import json
import hdfmap
import hdfmap.hdf_loader
from pytest import approx

from . import only_dls_file_system

# Edge case files, create this list from create_test_files.py
TEST_FILES = path.join(path.dirname(__file__), 'data', 'test_files.json')
with open(TEST_FILES, 'r') as f:
    CHECK_FILES = json.load(f)


@only_dls_file_system
def test_edge_cases():
    n = 0
    for chk in CHECK_FILES:
        print(f"Checking {chk['filename']}")
        assert path.isfile(chk['filename']) is True, f"{chk['filename']} doesn't exist"
        mymap = hdfmap.create_nexus_map(chk['filename'])
        assert isinstance(mymap, hdfmap.NexusMap), f"{chk['filename']} is not NexusMap"
        assert mymap('filepath') == chk['filename']
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
def test_very_old_i16_file():
    """file with no default plotting"""
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/777777.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)

    axes_names, signal_names = mymap.nexus_default_names()
    assert axes_names == {'TimeFromEpoch': '/entry1/measurement/TimeFromEpoch'}
    assert signal_names == {'sum': '/entry1/measurement/sum'}

    assert mymap['axes'] == '/entry1/measurement/TimeFromEpoch'
    assert mymap['signal'] == '/entry1/measurement/sum'

    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        data = mymap.get_plot_data(hdf)
        assert data['xlabel'] == 'TimeFromEpoch'
        assert data['ylabel'] == 'sum'
        assert data['xdata'].shape == (81, )
        assert data['ydata'].shape == (81,)
        assert data['axes_names'] == ['TimeFromEpoch']
        assert data['signal_names'] == ['sum']


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
def test_newer_i16_file():
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1109527.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        axes, signal, IMAGE, fname, fpath = mymap.eval(hdf, 'axes, signal, _IMAGE, filename, filepath')
    assert axes.shape == (61, ), 'expression "axes" has wrong shape'
    assert signal.shape == (61,), 'expression "axes" has wrong shape'
    assert signal.max() == 692919
    assert isinstance(fname, str), "expression 'filename' has wrong type"
    assert filename.endswith(fname)
    assert isinstance(fpath, str), "expression 'filepath' has wrong type"
    assert fpath == filename


@only_dls_file_system
def test_i16_bpm_file():
    """Tests an i16 scan with bpm images which are returned in the old style as TIFF"""
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1113658.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    assert mymap.get_image_path() == '/entry/instrument/bpm/path'
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        image = mymap.get_image(hdf)
    assert image.ndim == 0, 'bpm image has wrong shape'
    assert int(image) == 11, 'bpm image has wrong value'


@only_dls_file_system
def test_i16_default_signal():
    mymap = hdfmap.create_nexus_map('/dls/science/groups/das/ExampleData/i16/azimuths/1108750.nxs')

    axes_names, signal_names = mymap.nexus_default_names()

    assert next(iter(axes_names)) == 'eta_fly_fly'
    assert next(iter(signal_names)) == 'mroi2_sum'
    assert axes_names == {'eta_fly_fly': '/entry/measurement/eta_fly_fly'}
    assert len(signal_names) == 6
    assert signal_names == {
        'mroi2_sum': '/entry/instrument/mroi2/mroi2_sum',
        'count_time': '/entry/instrument/merlin/count_time',
        'merlin_max_val': '/entry/instrument/merlin/merlin_max_val',
        'merlin_max_x': '/entry/instrument/merlin/merlin_max_x',
        'merlin_max_y': '/entry/instrument/merlin/merlin_max_y',
        'merlin_total': '/entry/instrument/merlin/merlin_total'
    }


@only_dls_file_system
def test_i10_scannables():
    mymap = hdfmap.create_nexus_map('/dls/science/groups/das/ExampleData/hdfmap_tests/i10/i10-1-28428.nxs')
    scan_fields = mymap.get_data(mymap.load_hdf(), 'scan_fields')
    assert len(mymap.scannables) == len(scan_fields)


@only_dls_file_system
def test_i16_burst_mode():
    # See https://github.com/DiamondLightSource/hdfmap/issues/29
    filename = '/dls/science/groups/das/ExampleData/i16/burstmode_i16-788/16.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    shape = mymap.scannables_shape()
    image_shape = mymap.get_image_shape()

    with mymap.load_hdf() as hdf:
        cmd = mymap.eval(hdf, 'scan_command')
        scan_shape = hdf[mymap.get_image_path()].shape
        image = mymap.get_image(hdf, 0)

    # Burst mode scans return stack of images
    assert scan_shape != shape + image_shape
    assert scan_shape == (10, 5, 960, 1280)
    assert image.ndim == 3


@only_dls_file_system
def test_msmapper_file():
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/processed/1098101_msmapper.nxs'
    assert path.isfile(filename) is True, f"{filename} doesn't exist"
    mymap = hdfmap.create_nexus_map(filename)
    assert mymap['unit_cell'] == '/entry0/sample/unit_cell', 'link to old file incorrect'
    with hdfmap.hdf_loader.load_hdf(filename) as hdf:
        a, b, c, alpha, beta, gamma = mymap.eval(hdf, 'unit_cell')
    assert gamma > 1.0, 'unit cell incorrect'

    # defaults in /processed but scan_fields causes scannables to use /entry0
    assert len(mymap.scannables) == 36
    assert mymap.scannables_shape() == (81, )
    axes_paths, signal_paths = mymap.nexus_default_paths()
    assert axes_paths == ['/processed/reciprocal_space/h-axis', '/processed/reciprocal_space/k-axis', '/processed/reciprocal_space/l-axis']
    assert signal_paths == ['/processed/reciprocal_space/volume', '/processed/reciprocal_space/weight']
    axes_names, signal_names = mymap.nexus_default_names()
    assert axes_names == {'beamOK': '/entry0/roi2/beamOK'}
    assert signal_names == {'theta': '/entry0/sample/transformations/theta'}

    # alternative file with /entry0, /processed, /analysis{@default}
    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/processed/1109527_msmapper.nxs'
    mymap = hdfmap.create_nexus_map(filename, default_entry_only=False)
    assert len(mymap.scannables) == 23
    assert mymap.scannables_shape() == (61,)
    axes_paths, signal_paths = mymap.nexus_default_paths()
    assert axes_paths == ['/analysis/l_axis/l']
    assert signal_paths == ['/analysis/l_axis/intensity', '/analysis/l_axis/fit']
    axes_names, signal_names = mymap.nexus_default_names()
    assert axes_names == {'eta_fly_fly': '/entry0/instrument/eta_fly_fly/value'}
    assert signal_names == {'pil_total': '/entry0/instrument/pil3_100k/pil_total', 'pil_max_y': '/entry0/instrument/pil3_100k/pil_max_y'}

    # Real defaults stored in /analysis, normally overridden by scan_fields
    mymap = hdfmap.create_nexus_map(filename, default_entry_only=True)
    assert len(mymap.scannables) == 3
    assert mymap.scannables_shape() == (1185,)
    axes_names, signal_names = mymap.nexus_default_names()
    assert axes_names == {'l': '/analysis/l_axis/l'}
    assert signal_names == {'intensity': '/analysis/l_axis/intensity', 'fit': '/analysis/l_axis/fit'}


@only_dls_file_system
def test_alternate_name_local_data():
    f1 = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1109527.nxs'
    f2 = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1113658.nxs'
    m = hdfmap.create_nexus_map(f1)

    with hdfmap.load_hdf(f1) as nxs:
        scan_command1 = m.get_data(nxs, 'scan_command')
        print(scan_command1)
        cmd1 = m.eval(nxs, '(cmd|scan_command)')
        assert scan_command1 != cmd1, 'cmd and scan_command should not be the same'
    with hdfmap.load_hdf(f2) as nxs:
        scan_command2 = m.get_data(nxs, 'scan_command')
        cmd2 = m.eval(nxs, '(cmd|scan_command)')
        assert scan_command2 == cmd2, 'cmd and scan_command should be the same'
    assert cmd1 != cmd2, 'cmd of both files should not be the same'


@only_dls_file_system
def test_i06_pol_scan():
    f = '/dls/science/groups/das/ExampleData/hdfmap_tests/i06/i06-384074.nxs'
    m = hdfmap.create_nexus_map(f)

    axes_paths, signal_paths = m.nexus_default_paths()
    axes_names, signal_names = m.nexus_default_names()

    assert len(axes_paths) == len(axes_names)
    assert axes_names == {'pol': '/entry/medipix/pol', 'ds': '/entry/medipix/ds'}
    assert signal_names == {'YDriver1_meanvalue': '/entry/instrument/medipix/YDriver1_meanvalue'}


@only_dls_file_system
def test_complex_eval():
    f = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1109527.nxs'
    m = hdfmap.create_nexus_map(f)

    m.add_named_expression(**{
        '_t': '(count_time|counttime|t?(1.0))',
        '_cmd': '(cmd|user_input_command|user_command|scan_command)',
        'cmd': '(cmd|user_input_command|user_command|scan_command)',
    })

    with m.load_hdf() as hdf:
        assert m.eval(hdf, '_cmd') == 'flyscancn eta_fly 0.005 61 pil3_100k 0.1 0.5 roi1 roi2'
        assert m.eval(hdf, 'max(signal / Transmission / (rc/300.) / _t)') == approx(1215483134.5953412)
        assert m.eval(hdf, 'cmd') == 'flyscancn eta_fly 0.005 61 pil3_100k 0.1 0.5 roi1 roi2'


@only_dls_file_system
def test_another_complex_eval():
    f = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1109527.nxs'
    m = hdfmap.create_nexus_map(f)
    m.add_named_expression(count_time='(count_time|counttime|t?(1.0))')

    assert m.merge_default_names('count_time?(0.5)') == 'count_time'

    with m.load_hdf() as hdf:
        assert sum(m.eval(hdf, 'count_time?(0.5)')) == approx(6.1)