import pytest
import os
import h5py
import hdfmap

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FILE_HKL = DATA_FOLDER + "/1049598.nxs"  # hkl scan, pilatus
FILE_TREE = DATA_FOLDER + '/1049598.tree'  # hdf_tree_string of file
FILE_NEW_NEXUS = DATA_FOLDER + '/1040323.nxs'  # new nexus format
FILE_3D_NEXUS = DATA_FOLDER + '/i06-353130.nxs'  # new nexus format


@pytest.fixture
def files():
    files = hdfmap.file_functions.list_files(DATA_FOLDER, extension='.nxs')
    yield files

    
def test_hdf_tree_string():
    default_string = hdfmap.hdf_tree_string(FILE_HKL)
    nolinks_string = hdfmap.hdf_tree_string(FILE_HKL, all_links=False)
    assert len(nolinks_string) < len(default_string), "tree string with no links should be shorter"
    tree_string = open(FILE_TREE).read()
    # ignore the first line as it includes absolute file path
    assert default_string.splitlines()[1:] == tree_string.splitlines()[1:], "tree string is different"


def test_hdf_tree_strings(files):
    for file in files:
        assert len(hdfmap.hdf_tree_string(file)) > 0, f"{file} has no tree"
        assert len(hdfmap.hdf_tree_string(file, all_links=False)) > 1000, f"{file} has no tree without links"
        assert len(hdfmap.hdf_tree_string(file, attributes=False)) > 1000, f"{file} has no tree without attributes"
        assert len(hdfmap.hdf_tree_string(file, all_links=False, attributes=False)) > 1000, f"{file} has no tree"
        assert len(hdfmap.hdf_tree_string(file, group='/')) > 1000, f"{file} has no entry"


def test_hdf_dataset_list():
    dataset_list = hdfmap.hdf_dataset_list(FILE_HKL)
    nolinks_list = hdfmap.hdf_dataset_list(FILE_HKL, all_links=False)
    assert len(nolinks_list) < len(dataset_list), "dataset list with no links should be shorter"
    assert len(dataset_list) == 360, "Incorrect number of dataset paths"
    assert len(nolinks_list) == 301, "Incorrect number of dataset paths"


def test_hdf_tree_dict():
    tree = hdfmap.hdf_tree_dict(FILE_HKL)
    assert 'entry1' in tree
    assert 'measurement' in tree['entry1']
    assert 'sum' in tree['entry1']['measurement']
    assert tree['entry1']['measurement']['sum'] == 'float64, (101,)'


def test_hdf_compare():
    comparison = hdfmap.hdf_compare(FILE_HKL, FILE_NEW_NEXUS)
    print(comparison)
    assert len(comparison) == 26879, "comparison string wrong length"


def test_hdf_find():
    group_paths, dataset_paths = hdfmap.hdf_find(FILE_HKL, 'NXdata')
    assert len(group_paths) == 4, "Should find 4 NXdata groups"
    assert len(dataset_paths) == 89, "Should find 89 datasets in NXdata groups"

    group_paths, dataset_paths = hdfmap.hdf_find(FILE_HKL, 'NXdata', 'sum')
    assert len(group_paths) == 0, "Should find no NXdata/sum groups"
    assert len(dataset_paths) == 2, "Should find 2 datasets of type NXdata/sum"


def test_hdf_find_first():
    path1 = hdfmap.hdf_find_first(FILE_NEW_NEXUS, 'NXentry', 'NXinstrument', 'NXdetector', 'data')
    path2 = hdfmap.hdf_find_first(FILE_NEW_NEXUS, 'NXslit', 'x_gap')
    path3 = hdfmap.hdf_find_first(FILE_NEW_NEXUS, 'measurement', 'h')
    assert path1 == 'entry/instrument/pil3_100k/data'
    assert path2 == 'entry/instrument/s1/x_gap'
    assert path3 == 'entry/measurement/h'
    with h5py.File(FILE_NEW_NEXUS, 'r') as hdf:
        assert path1 in hdf, "NXdetector:data not found in file"
        assert path2 in hdf, "NXslit:x_gap not found in file"
        assert path3 in hdf, "measurement:h not found in file"


def test_hdf_linked_files():
    linked_files = hdfmap.hdf_linked_files(FILE_3D_NEXUS)
    assert len(linked_files) == 1, "Wrong number of linked files found"
    for file in linked_files:
        file_path = os.path.join(os.path.dirname(FILE_3D_NEXUS), file)
        assert os.path.isfile(file_path), f"'{file}' does not exist"

