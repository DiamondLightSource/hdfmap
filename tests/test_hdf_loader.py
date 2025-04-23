import pytest
import os
import hdfmap.file_functions as ff
import hdfmap.hdf_loader

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def files():
    files = ff.list_files(DATA_FOLDER, extension='.nxs')
    yield files


def test_create_hdf_map(files):
    for file in files:
        assert len(hdfmap.hdf_tree_string(file)) > 0, f"{file} has no tree"
        assert len(hdfmap.hdf_tree_string(file, all_links=False)) > 1000, f"{file} has no tree without links"
        assert len(hdfmap.hdf_tree_string(file, attributes=False)) > 1000, f"{file} has no tree without attributes"
        assert len(hdfmap.hdf_tree_string(file, all_links=False, attributes=False)) > 1000, f"{file} has no tree"
        assert len(hdfmap.hdf_tree_string(file, group='/')) > 1000, f"{file} has no entry"

