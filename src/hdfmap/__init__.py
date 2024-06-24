"""

"""


from .hdfmap_class import HdfMap
from .nexus import NexusMap
from .file_functions import load_hdf, create_hdf_map, create_nexus_map
from .file_functions import multifile_get_data, multifile_eval, multifile_format
from .reloader_class import HdfReloader


__version__ = "0.1.0"
__date__ = "2024/06/24"


def version_info():
    return 'hdfmap version %s (%s)' % (__version__, __date__)


def module_info():
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
    try:
        import hdf5plugin
        out += '\n    hdf5plugin: %s' % hdf5plugin.version
    except ImportError:
        out += '\n    hdf5plugin: None'
    import os
    out += '\nRunning in directory: %s\n' % os.path.abspath('.')
    return out
