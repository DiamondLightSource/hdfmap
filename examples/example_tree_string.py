"""
hdfmap example

Print the HDF file structure

By Dan Porter
23/04/2025 (hdfmap v0.9.0)
"""

import hdfmap

if __name__ == '__main__':
    filename = '../tests/data/i06-353130.nxs'

    print(hdfmap.hdf_tree_string(filename, all_links=True, attributes=True))
