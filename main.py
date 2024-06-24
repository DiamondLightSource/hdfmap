"""
Test hdfmap
"""

import hdfmap


hdf_file = "/scratch/grp66007/data/i16/2024/mm36462-1/1049598.nxs"

mymap = hdfmap.create_nexus_map(hdf_file, groups=['measurement'])
rdr = hdfmap.HdfReloader(hdf_file, hdf_map=mymap)

print(mymap)
