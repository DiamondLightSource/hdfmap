"""
Test hdfmap
"""

import os
import numpy as np
import h5py
import hdfmap
from hdfmap.file_functions import list_files
from hdfmap.dataloader_class import create_nexus_datafile


datadir = '/scratch/grp66007/data/i16/2024/cm37262-9'
hdf_file = "/scratch/grp66007/data/i16/2024/mm36462-1/1049598.nxs"  # hkl scan, pilatus
nxs_file = "/scratch/grp66007/data/i16/example/1040323.nxs"  # new nexus format

nxs_map = hdfmap.create_nexus_map(nxs_file, groups=['instrument', 'sample'])
mymap = hdfmap.create_nexus_map(hdf_file, groups=['before_scan', 'instrument', 'measurement', 'sample'])
rdr = hdfmap.HdfReloader(hdf_file, hdf_map=mymap)

print(mymap)
print(mymap.info_scannables())

[data1, data2] = rdr.get_data(['scan_command', 'start_time'])
print(data1, data2)
print(rdr.eval('h[len(h)//2]'))
print(rdr.format('The energy is {en} keV'))


nxs_files = list_files(datadir)
cmd = hdfmap.multifile_get_data('scan_command', nxs_files, debug=True)

print(f"\n\n{datadir}")
print('\n'.join(f"{os.path.basename(f):>20}: {c}" for f, c in zip(nxs_files, cmd)))

print(f"\n\nFile: {nxs_file}\nScannables:")
with h5py.File(nxs_file, 'r') as nxs:
    print(nxs_map.get_scannables_str(nxs))

print(f"\n\nCreate DataObject: {nxs_file}")
d = create_nexus_datafile(nxs_file)
print(d.metadata.scan_command)
