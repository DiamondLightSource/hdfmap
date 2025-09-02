"""
hdfmap
Map objects within an HDF5 file and create a dataset namespace.

# Usage
### HdfMap from NeXus file
    from hdfmap import create_nexus_map, load_hdf
    hmap = create_nexus_map('file.nxs')
    with load_hdf('file.nxs') as nxs:
        address = hmap.get_address('energy')
        energy = nxs[address][()]
        string = hmap.format_hdf(nxs, "the energy is {energy:.2f} keV")
        d = hmap.get_dataholder(nxs)  # classic data table, d.scannable, d.metadata

### Shortcuts - single file reloading class
    from hdfmap import NexusLoader
    scan = NexusLoader('file.nxs')
    [data1, data2] = scan.get_data(['dataset_name_1', 'dataset_name_2'])
    data = scan.eval('dataset_name_1 * 100 + 2')
    string = scan.format('my data is {dataset_name_1:.2f}')

### Shortcuts - multifile load data
    from hdfmap import hdf_data, hdf_eval, hdf_format, hdf_image
    all_data = hdf_data([f"file{n}.nxs" for n in range(100)], 'dataset_name')
    normalised_data = hdf_eval(filenames, 'total / Transmission / (rc / 300.)')
    descriptions = hdf_eval(filenames, 'Energy: {en:5.3f} keV')
    image = hdf_image(filenames, index=31)


Copyright 2024-2025 Daniel G. Porter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


By Dr Dan Porter
Diamond Light Source Ltd
2024-2025
"""

from .logging import set_all_logging_level
from .hdf_loader import (load_hdf, hdf_tree_string, hdf_tree_dict, hdf_dataset_list, hdf_compare, hdf_find,
                         hdf_find_first, hdf_linked_files)
from .hdfmap_class import HdfMap
from .nexus import NexusMap
from .file_functions import list_files, create_hdf_map, create_nexus_map
from .file_functions import hdf_data, hdf_image, hdf_eval, hdf_format, nexus_data_block, compare_maps
from .reloader_class import HdfLoader, NexusLoader


__all__ = [
    'load_hdf', 'create_hdf_map', 'create_nexus_map', 'list_files',
    'hdf_tree_string', 'hdf_tree_dict', 'hdf_dataset_list', 'hdf_compare', 'compare_maps', 'hdf_find',
    'hdf_find_first', 'hdf_linked_files',
    'hdf_data', 'hdf_image', 'hdf_eval', 'hdf_format', 'nexus_data_block', 'HdfLoader', 'NexusLoader',
    'set_all_logging_level', 'version_info', 'module_info'
]

__version__ = "1.0.1"
__date__ = "2025/09/02"


def version_info() -> str:
    return 'hdfmap version %s (%s)' % (__version__, __date__)


def module_info() -> str:
    import sys
    out = 'Python version %s' % sys.version
    out += '\n at: %s' % sys.executable
    out += '\n %s: %s' % (version_info(), __file__)
    # Modules
    import numpy
    out += '\n     numpy version: %s' % numpy.__version__
    import h5py
    out += '\n      h5py version: %s' % h5py.__version__
    # import imageio
    # out += '\n   imageio version: %s' % imageio.__version__
    import asteval
    out += '\n   asteval version: %s' % asteval.__version__
    try:
        import hdf5plugin
        out += '\n    hdf5plugin: %s' % hdf5plugin.version
    except ImportError:
        out += '\n    hdf5plugin: None'
    import os
    out += '\nRunning in directory: %s\n' % os.path.abspath('.')
    return out
