
import h5py

try:
    import hdf5plugin  # required for compressed data
except ImportError:
    print('Warning: hdf5plugin not available.')


def load_hdf(hdf_filename: str) -> h5py.File:
    """Load hdf file, return h5py.File object"""
    return h5py.File(hdf_filename, 'r')
