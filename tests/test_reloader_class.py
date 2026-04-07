import os
from hdfmap import HdfLoader, NexusLoader

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_NEW_NEXUS = DATA_FOLDER + '/1040323.nxs'  # new nexus format
FILE_3D_NEXUS = DATA_FOLDER + '/i06-353130.nxs'  # new nexus format


def test_hdf_reloader():
    scan = HdfLoader(FILE_NEW_NEXUS)

    assert len(scan('h')) == 21
    cmd = scan.format('{(cmd|scan_command)}')
    assert cmd == 'scan hkl [0.97, 0.022, 0.112] [0.97, 0.022, 0.132] [0, 0, 0.001] MapperProc pil3_100k 1'

    path = scan.get_hdf_path('TimeSec')
    assert path == '/entry/measurement/TimeSec'

    h_vals, k_vals, l_vals = scan.get_data('h', 'k', 'l')
    assert len(h_vals) == len(k_vals) == len(l_vals)

    # Local data
    scan.add_local(my_data=55, more_data='hello', scan_command='overwriting')
    assert scan('my_data') == 55
    assert scan('my_data, more_data') == (55, 'hello')
    assert 'my_data' in scan
    assert scan('scan_command') == 'overwriting'
    assert scan.eval('scan_command', prefer_local=False) == cmd


def test_nexus_reloader():
    scan = NexusLoader(FILE_NEW_NEXUS)

    assert len(scan('axes')) == 21
    assert len(scan('signal')) == 21
    assert scan('fast_pixel_direction@units') == 'mm'

    data = scan.get_plot_data()
    KEYS = {'xlabel', 'ylabel', 'xdata', 'ydata', 'axes_names', 'signal_names', 'axes_data',
            'signal_data', 'axes_labels', 'signal_labels', 'data', 'title'}
    assert data.keys() >= KEYS

