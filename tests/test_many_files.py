import os
from time import perf_counter
import hdfmap

from . import only_dls_file_system

# Folder with over 1000 files
THOUSAND_FILES = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/cm37262-1'
NFILES = 1332  # number of files to test (max 1332)
# FORMAT_STRING = """#{entry_identifier}\ncmd: {scan_command}\n"""
FORMAT_STRING = '#{entry_identifier}: {start_time} : E={incident_energy:.3f} keV : {scan_command}'

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
LOCAL_FILES = [
    DATA_FOLDER + '/1040323.nxs',  # new nexus format
    DATA_FOLDER + '/i06-353130.nxs',  # new nexus format
]


def test_use_local_data():
    m = hdfmap.create_nexus_map(LOCAL_FILES[0])
    exp = "(cmd|scan_command)"
    cmd = [m.eval(hdfmap.load_hdf(f), exp) for f in LOCAL_FILES]
    assert len(set(cmd)) > 1, "datasets from different files are the same when they shouldn't be"
    m.use_local_data(True)
    cmd = [m.eval(hdfmap.load_hdf(f), exp) for f in LOCAL_FILES]
    assert len(set(cmd)) == 1, "datasets from different files are different when they shouldn't be"


def test_population_time():
    # time to populate a single file
    # Use the i06 file as the i16 file is very large
    n_tries = 10
    population_time = [0, 0]
    for n, filename in enumerate(LOCAL_FILES):
        with hdfmap.load_hdf(filename) as hdf:
            mymap = hdfmap.NexusMap()
            mymap.populate(hdf)  # warm-up
            start = perf_counter()
            for _ in range(n_tries):
                mymap.populate(hdf)
            stop = perf_counter()
        population_time[n] = (stop - start) / n_tries
        print(f"Population time ('{os.path.basename(filename)}'): {population_time[n]:.3f} s")
    print(f"Large i16 file takes {population_time[0] / population_time[1]:.2%} longer than i06 file")
    assert population_time[0] / population_time[1] < 5


@only_dls_file_system
def test_compare_time_for_many_files():
    files = hdfmap.list_files(THOUSAND_FILES)[:NFILES]
    assert len(files) > 1, "Files not found"
    # time to read single entry from each files
    start = perf_counter()
    output1 = []
    for file in files:
        with hdfmap.load_hdf(file) as hdf:
            output1.append((
                hdf['/entry1/scan_command'][()],
                hdf['/entry1/entry_identifier'][()],
                hdf['/entry1/start_time'][()],
                hdf['/entry1/sample/beam/incident_energy'][()],
            ))
    stop = perf_counter()
    single_time = stop - start

    # time to read multiple entries using hdfmap
    start = perf_counter()
    output2 = hdfmap.hdf_format(files, FORMAT_STRING)
    stop = perf_counter()
    multi_time = stop - start

    # Repeat direct load to avoid file initilisation errors
    start = perf_counter()
    output1 = []
    for file in files:
        with hdfmap.load_hdf(file) as hdf:
            output1.append((
                hdf['/entry1/scan_command'][()],
                hdf['/entry1/entry_identifier'][()],
                hdf['/entry1/start_time'][()],
                hdf['/entry1/sample/beam/incident_energy'][()],
            ))
    stop = perf_counter()
    single_time2 = stop - start

    print(f"\nRead single entry from {len(files)} files in: {single_time:.3f} s")
    print(f"Read multi entry from {len(files)} files in: {multi_time:.3f} s")
    print(f"Read single entry from {len(files)} files in: {single_time2:.3f} s")
    print(f"Performance factor: {((multi_time-single_time2) / single_time2):+.1%} of direct access time")
    # Typically around 40% slower
    assert multi_time < 1.5 * single_time2, "mult-read is much slower than direct read"


@only_dls_file_system
def test_local_data_from_different_files():
    """Test for issue #42 - multiple instances of HdfLoader contain the same local namespace"""
    # i21 files 2026 mm23841-1
    folder = '/dls/science/groups/das/ExampleData/hdfmap_tests/i21/mm23841-1'
    file1, file2 = [folder + f'/i21-{n}.nxs' for n in [472812, 472811]]
    m1 = hdfmap.create_nexus_map(file1)
    m2 = hdfmap.create_nexus_map(file2)
    assert 'user_input_command' in m1
    assert 'user_input_command' not in m2

    scan1 = hdfmap.NexusLoader(file1, m1)
    scan2 = hdfmap.NexusLoader(file2, m1)
    scan1.add_local(some_data=55)
    assert scan1.map is scan2.map

    cmd1 = scan1.format('{(cmd|user_input_command|user_command|scan_command)}')
    cmd2 = scan2.format('{(cmd|user_input_command|user_command|scan_command)}')
    assert cmd1 != cmd2
    assert scan1('some_data') == 55

    # setting local data of scan2 also updates scan 1
    scan2.add_local(some_data=44)
    assert scan2('some_data') == 44
    assert scan1('some_data') == 55
