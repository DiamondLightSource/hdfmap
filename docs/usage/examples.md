# Examples

### scan data & metadata
Separate datasets in a NeXus file into Diamond's classic scannables and metadata, similar to what was in the old
'*.dat' files.

```python
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file:
hmap = create_nexus_map('file.nxs')
with load_hdf('file.nxs') as nxs:
    scannables = hmap.get_scannables_array(nxs)  # creates 2D numpy array
    labels = scannables.dtype.names
    metadata = hmap.get_metadata(nxs)  # {'name': value}
    d = hmap.get_dataholder(nxs)  # classic data table, d.scannable, d.metadata
d.theta == d['theta']  # scannable array 'theta'
d.metadata.scan_command == d.metadata['scan_command']  # single value 'scan_command'

# OR, use the shortcut:
from hdfmap import nexus_data_block

d = nexus_data_block('file.nxs')

# The data loader class removes the need to open the files:
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')
metadata = scan.get_metadata()
scannables = scan.get_scannables()
```


### evaluating expressions using hdf data
String expressions can be evaluated accessing datasets within the file using only their names, 
without needing to know their full location in the file.

```python
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')

value = scan.eval('incident_energy')  # returns value associated with /entry/group/incident_energy
norm = scan.eval('np.max(max_val) / 100')  # expressions can include operations, includes numpy as np
data = scan.eval('NXdetector_data[:, 50:60, 60:60]')  # array expressions, group_name and class_name are allowed
axes, signal = scan('axes, signal') # NeXus default signal and axes are in the namespace, scan() is a shortcut
```

#### Rules for names in eval/format spec:
 - 'name' - returns value of dataset '/entry/group/name'
 - 'group_name' - return value of dataset '/entry/group/name'
 - 'class_name' - return value of dataset '/entry/group/name' where group has NXclass: class
 - 'name@attr' - returns attribute 'attr' associated with dataset 'name'
 - '_name' - retrun hdf path of dataset 'name'
 - '__name' - return default name of dataset 'name' (used when requesting 'axes' or 'signal'
 - 'filename', 'filepath' - these are always available



### formatted strings from metadata
Format strings can also be parsed to obtain data from the hdf files. 
names inside {} are names of datasets within the hdf file and will return values associated with this. 
Otherwise, formatting rules for {} are the same as for standard python f-strings.

```python
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')

string = scan.format('axes: {__axes0} = {axes0}')
```

#### multi-file example
Generate single line strings from many files very quickly, loading only the datasets required from the file.
```python
from hdfmap import hdf_format

list_of_files = [f"file{n}.nxs" for n in range(1000)]
fmt = "{filename:20}: {incident_energy:6.2f} {incident_energy@units} : {scan_command}"

output_str = hdf_format(list_of_files, fmt)  # the hdfmap is generated for the first file, then the same paths are used
```

### automatic default plot axes
If defined in the nexus file, 'axes' and 'signal' will be populated automatically

```python
import matplotlib.pyplot as plt
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file:
hmap = create_nexus_map('file.nxs')
with load_hdf('file.nxs') as nxs:
    axes = hmap.get_data(nxs, 'axes')
    signal = hmap.get_data(nxs, 'signal')
    title = hmap.format_hdf(nxs, "{entry_identifier}\n{scan_command}")
axes_label = hmap.get_path('axes')
signal_label = hmap.get_path('signal')
# plot the data (e.g. using matplotlib)
plt.figure()
plt.plot(axes, signal)
plt.xlabel(axes_label)
plt.ylabel(signal_label)
plt.title(title)

# Or, using NexusLoader:
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')
axes, signal = scan('axes, signal')
axes_label, signal_label = scan('_axes, _signal')
title = scan.format("{entry_identifier}\n{scan_command}")
```

### Automatic image data
Get images from the first detector in a NeXus file

```python
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file:
hmap = create_nexus_map('file.nxs')
image_location = hmap.get_image_path()  # returns the hdf path chosen for the default detector
with load_hdf('file.nxs') as nxs:
    middle_image = hmap.get_image(nxs)  # returns single image from index len(dataset)//2
    first_image = hmap.get_image(nxs, 0)  # returns single image from dataset[0, :, :]
    volume = hmap.get_image(nxs, ())  # returns whole volume as array
    roi = hmap.get_image(nxs, (0, slice(5, 10, 1), slice(5, 10, 1)))  # returns part of dataset

# Or, using NexusLoader:
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')
image = scan.get_image(index=0)  # using index as defined above
```

### Multi-scan metadata string
Generate a metadata string from every file in a directory very quickly. The HdfMap is only created for the first file,
the remaining files are treated as having identical structure.
```python
from hdfmap import list_files, hdf_format

format_string = "#{entry_identifier}: {start_time} : E={incident_energy:.3f} keV : {scan_command}"
files = list_files('/directoy/path', extension='.nxs')
strings_list = hdf_format(files, format_string)
print('\n'.join(strings_list))

# other multi-file readers:
from hdfmap import hdf_data, hdf_image, hdf_eval

data_list = hdf_data(files, 'incident_energy')
image_list = hdf_image(files, index=0)
data_list = hdf_eval(files, 'signal / Transmission')
```

### Multi-Dimension Scans
Where files have multiple axes (e.g. a scan of x and y), the NeXus file will have a list of default @axes arguments.
The reserved name 'axes' only gives the first axis, but additional axes paths are stored as names 'axes#'
where # is the axis number.
```python
from hdfmap import NexusLoader

scan = NexusLoader('scan_file.nxs')
axes_paths, signal_path = scan.map.nexus_defaults()

axes_x = scan('axes0')
axes_y = scan('axes1')
signal = scan('signal')

# Image data from MD scans
scan_shape = scan.map.get_image_shape()
index_slice = scan.map.get_image_index(30)  # returns (i,j,k) 
```

### General plot data
Shortcuts for producing default plots of data
```python
import matplotlib.pyplot as plt
from hdfmap import NexusLoader

scan = NexusLoader('scan_file.nxs')

data = scan.get_plot_data() 
data = {
    'xlabel': '',  # str label of first axes
    'ylabel': '',  # str label of signal
    'xdata': [],  # flattened array of first axes
    'ydata': [],  # flattend array of signal
    'axes_names': [''],  # list of axes names,
    'signal_name': '',  # str signal name,
    'axes_data': [[], ],  # list of ND arrays of data for axes,
    'signal_data': [],  # ND array of signal data,
    'data': {'': []},  # dict of all scannables axes,
    'axes_labels': [''],  # list of axes labels as 'name [units]',
    'title': '',  # str title as 'filename\nNXtitle'
}

fig, ax = plt.subplot()
ax.plot(data['xdata'], data['ydata'], label=data['signal_name'])
ax.set_xlabel(data['xlabel'])
ax.set_ylabel(data['ylabel'])
ax.set_title(data['title'])
fig.show()
```


