"""
Test hdfmap
"""

import hdfmap


hdf_file = "/scratch/grp66007/data/i16/2024/mm36462-1/1049598.nxs"

mymap = hdfmap.create_nexus_map(hdf_file, groups=['measurement'])
rdr = hdfmap.HdfReloader(hdf_file, hdf_map=mymap)

print(mymap)

[data1, data2] = rdr.get_data(['scan_command', 'start_time'])
print(data1, data2)
print(rdr.eval('h[len(h)//2]'))
print(rdr.format('The energy is {en} keV'))

#all_data = hdfmap.multifile_get_data('dataset_name', [f'file{n}.nxs' for n in range(100)])


