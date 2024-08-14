import os
from time import perf_counter
import hdfmap


# Folder with over 1000 files
THOUSAND_FILES = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/cm37262-1'
NFILES = 1332  # number of files to test (max 1332)
# FORMAT_STRING = """#{entry_identifier}\ncmd: {scan_command}\n"""
FORMAT_STRING = '#{entry_identifier}: {start_time} : E={incident_energy:.3f} keV : {scan_command}'


def test_compare_time_for_many_files():
    files = hdfmap.list_files(THOUSAND_FILES)[:NFILES]
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

    print(f"\nRead single entry from {len(files)} files in: {single_time:.3f} s")
    print(f"Read multi entry from {len(files)} files in: {multi_time:.3f} s")
    print(f"Performance factor: {((multi_time-single_time) / single_time):+.1%} of direct access time")
    # Typically around 40% slower
    assert multi_time < 1.5 * single_time, "mult-read is much slower than direct read"
