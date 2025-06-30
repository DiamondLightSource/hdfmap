"""
hdfmap Example
"""

import hdfmap

nxs_file = '../tests/data/1049598.nxs'
new_file = '../tests/data/1040323.nxs'

new_map = hdfmap.create_nexus_map(new_file)
nxs_map = hdfmap.create_nexus_map(nxs_file)
print(nxs_map)
print(nxs_map.info_names(scannables=True))

rdr = hdfmap.NexusLoader(nxs_file, nxs_map)
[data1, data2] = rdr.get_data(*['scan_command', 'start_time'])
print(data1, data2)
print(rdr.eval('h[len(h)//2]'))
print(rdr.format('The energy is {en} keV'))

print(f"\n\nFile: {nxs_file}\nScannables:")
with hdfmap.hdf_loader.load_hdf(nxs_file) as nxs:
    print(nxs_map.create_metadata_list(nxs))
    print(nxs_map.create_scannables_table(nxs))

print(f"\n\nCreate DataObject: {nxs_file}")
with hdfmap.hdf_loader.load_hdf(nxs_file) as nxs:
    d = nxs_map.get_dataholder(nxs)
print(d.metadata.scan_command)

# THOUSAND_FILES = '/dls/science/groups/das/ExampleData/hdfmap_tests/i16/cm37262-1'
# files = hdfmap.list_files(THOUSAND_FILES)
# format_string = '#{entry_identifier}: {start_time} : E={incident_energy:.3f} keV : {scan_command}'
# strings = hdfmap.hdf_format(files, format_string)


