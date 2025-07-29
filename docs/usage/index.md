# Usage
HdfMap is a python module. It is designed for use in python interactive terminals or scripts. 
The idea is that you can easily read data from HDF or NeXus files without needed to know the HDF paths or file 
schema.

```python
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file:
m = create_nexus_map('file.nxs')
m['energy']  # >> '/entry/instrument/monochromator/energy'
m['signal']  # >> '/entry/measurement/sum'
m['axes']  # >> '/entry/measurement/theta'
m.get_image_path()  # >> '/entry/instrument/pil3_100k/data'

with load_hdf('file.nxs') as nxs:
    path = m.get_path('scan_command')
    cmd = nxs[path][()]  # returns bytes data direct from file
    cmd = m.get_data(nxs, 'scan_command')  # returns converted str output
    string = m.format_hdf(nxs, "the energy is {energy:.2f} keV")
    image = m.get_image(nxs, 30)  # returns image number 30 in list (even in multi-dimension)
    d = m.get_dataholder(nxs)  # classic data table, d.scannable, d.metadata

# new in V1.0.0 - evaluate name based expressions in original file
m('signal / count_time') # >> numpy array

# Shortcuts - single file reloader class
from hdfmap import NexusLoader

scan = NexusLoader('file.hdf')
[data1, data2] = scan.get_data(['dataset_name_1', 'dataset_name_2'])
data = scan.eval('dataset_name_1 * 100 + 2')
string = scan.format('my data is {dataset_name_1:.2f}')
data1, data2 = scan('dataset_name_1, dataset_name_2')
path1, path2 = scan('_dataset_name_1, _dataset_name_2')
xdata, xlabel = scan('axes0, _axes0')
ydata, ylabel = scan('signal, _signal')

# Shortcuts - multifile load data (generate map from first file)
from hdfmap import hdf_data, hdf_eval, hdf_format, hdf_image

all_data = hdf_data([f'file{n}.nxs' for n in range(100)], 'dataset_name')
normalised_data = hdf_eval(filenames, 'total / Transmission / (rc / 300.)')
descriptions = hdf_format(filenames, 'Energy: {en:5.3f} keV')
image_stack = hdf_image(filenames, index=31)
```