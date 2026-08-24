"""
Tests for downstream software
"""

import sys
import os
from . import only_dls_file_system


@only_dls_file_system
def test_nexus2srs():
    sys.path.append('/dls_sw/apps/nexus2srs/1.1.1/nexus2srs/')
    from nexus2srs import nxs2dat

    filename = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/1109527.nxs'
    nxs2dat(filename, 'data/check_1109527.dat')

    with open('data/check_1109527.dat', 'r') as new, open('data/1109527.dat', 'r') as old:
        old_text = old.read()
        new_text = new.read()
        assert old_text == new_text

    os.remove('data/check_1109527.dat')