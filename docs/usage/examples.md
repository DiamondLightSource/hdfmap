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

Expressions are parsed in a two-step process. The first step identifies symbols in the expression 
that are available in the file, replaces patterns in the expression and builds a data namespace with data from the file. 
The seconds step evaluates the modified expression in the data namespace.

#### Rules for names in eval/format spec:
 - 'filename', 'filepath' - these are always available
 - 'name' - returns value of dataset '/entry/group/name'
 - 'group_name' - return value of dataset '/entry/group/name'
 - 'class_name' - return value of dataset '/entry/group/name' where group has NXclass: class
 - 'name@attr' - returns attribute 'attr' associated with dataset 'name'
 - '_name' - return hdf path of dataset 'name'
 - '__name' - return default name of dataset 'name' (used when requesting 'axes' or 'signal')
 - 's_*name*': string representation of dataset (includes units if available)
 - 'd_*name*': return dataset object. **warning**: this may result in the hdf file or external files not closing on completion
 - '*name*@*attr*': returns attribute of dataset *name*
 - '*name*?(*default*)': returns default if *name* doesn't exist
 - '(name1|name2|name3)': returns the first available of the names
 - '(name1|name2?(default))': returns the first available name or default


#### New in V0.8.1: local variables in eval/format
Additional variables can be assigned to the local namespace accessed during eval or format, either directly accessing
data, or as shorthand for a path or expression.

**Note V1.0.2**: local variables will be overwritten by values from files, unless use_local_data() is set.   

```python
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')

# add local data
scan.map.add_local(my_parameter=800.)
monitor = scan.eval('ic1monitor / my_parameter')
# add replacement path
scan.map.add_named_expression(cmd='/entry1/scan_command')
cmd = scan.eval('cmd')
# add short-hand expressions
expr = {
    'cmd': 'scan_command',
    'normby': 'Transmission/count_time/(ic1monitor/800.)',
}
scan.map.add_named_expression(**expr)
ydata = scan.eval('signal/normby')
```

#### New in V1.0.2: reload data from local namespace
When datasets are read using hdfmap.eval or similar, the data is stored with local data. This data can be accessed 
rapidly by turning the option on. This is particuarly useful when having to reload large datasets over a slow 
network connection, but should not be used if reading from multiple files using the same hdfmap object.

```python
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')
scan.map.use_local_data()

volume = scan.eval('IMAGE')  # loads the entire image volume, which can be slow
norm_vol = scan.eval('IMAGE / Transmission')  # repeated call is much faster because IMAGE was already in memory

scan.map.use_local_data(False)  # return to the default behaviour
```

#### New in V1.0.0: load datasets
An additional pattern `'d_*name*'` has been added, allowing hdf dataset objects to be returned. This allows for lazy
indexing of large datasets, however returning the dataset locally could result in the hdf file, or external files
pointed to by the dataset, remaining open.

Also, an additional behaviour has been added allowing `HdfMap('expression')`, which opens the default hdf file as a
shorthand for `HdfMap.eval(HdfMap.load_hdf(), 'expression')`.

```python
from hdfmap import create_nexus_map

m = create_nexus_map('file.nxs')

dataset = m('d_incident_energy') # -> returns h5py.Dataset object, note that file.nxs is now open
del dataset # removing the link will close file.nxs

# load detector image (stored in external file)
image_dataset = m('d_IMAGE')  # -> returns h5py.Dataset
# note that file.nxs will be closed as the dataset points to an external file, however this external file will be open

# Lazily load just the region on the detector (files will be closed as dataset reference lost)
roi = m('d_IMAGE[..., 90:110, 200:240]')  # -> array with shape (N,20,40).  
roi_sum = m('d_IMAGE[..., 90:110, 200:240].sum(axis=(-1, -2))') # -> array with shape (N)
```

Note: External datasets are currently not closed by context managers (i.e. `with h5py.File...`). 
This behaviour is a [known bug](https://github.com/h5py/h5py/issues/2454) and may be fixed in the future.

### New in V1.0.1: Custom ROIs
User defined Regions of Interest (ROIs) can be assigned for image datasets. The ROIs will only
be evaluated when read and will only read the required region - not the whole dataset.

On defining a ROI, several names will be added to the namespace:
 - *name* -> returns the whole ROI array as a HDF5 dataset
 - *name*_total -> returns the sum of each image in the ROI array
 - *name*_max -> returns the max of each image in the ROI array
 - *name*_min -> returns the min of each image in the ROI array
 - *name*_mean -> returns the mean of each image in the ROI array
 - *name*_bkg -> returns the background ROI array (area around ROI)
 - *name*_rmbkg -> returns the total with background subtracted
 - *name*_box -> returns the pixel positions of the ROI
 - *name*_bkg_box -> returns the pixel positions of the background ROI

Each ROI is given a name and must have a centre and widths defined. The centre value can be a string expression,
allowing you to call other values from the file, such as a defined detector centre.

```python
from hdfmap import create_nexus_map

m = create_nexus_map('file.nxs')
m.add_roi('my_roi1', 'pil3_centre_j', 'pil3_centre_i', 31, 21)
roi_volume = m('my_roi1')  # shape: (n, 31, 21)
roi_total = m('my_roi1_total')  # shape: (n, )
```


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
axes_paths, signal_paths = scan.map.nexus_default_paths()

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

### Metadata Evaluation 
Functionality for namespace evaluation of the hdf file allows for a number of rules allowing easy extraction
of formatted metadata. The Evaluation functions are:

 - `HdfMap.eval(hdfobj, 'name')` -> value
 - `HdfMap.format_hdf(hdfobj, '{name}')` -> string
 - `HdfLoader('eval')` -> value
 - `HdfLoader.eval('eval')` -> value
 - `HdfLoader.format('{name}')` -> string
 - `hdf_eval([files], 'name')` -> list[values]
 - `hdf_format([files], '{name}')` -> list[string]

#### eval vs format
Evaluation functions evaluate the expression as given, replacing names in the hdfmap namespace with their associated
values, or using the rules below. The format functions allow the input of python 
[f-strings](https://docs.python.org/3/tutorial/inputoutput.html#fancier-output-formatting),
allowing precise formatting to be applied and returning a string.

#### Rules
The following patterns are allowed in any expression:
 - 'filename': str, name of hdf_file
 - 'filepath': str, full path of hdf_file
 - '_*name*': str hdf path of *name*
 - '__*name*': str internal name of *name* (e.g. for 'axes')
 - 's_*name*': string representation of dataset (includes units if available)
 - '*name*@attr': returns attribute of dataset *name*
 - '*name*?(default)': returns default if *name* doesn't exist
 - '(name1|name2|name3)': returns the first available of the names
 - '(name1|name2@(default))': returns the first available name or default

#### Examples
```python
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file:
hmap = create_nexus_map('file.nxs')
with load_hdf('file.nxs') as nxs:
    # mathematical array expressions (using np as Numpy)
    data = hmap.eval(nxs, 'int(np.max(total / Transmission / count_time))')
    # return the path of a name
    path = hmap.eval(nxs, '_axes')  # -> '/entry/measurement/h'
    # return the real name of a variable
    name = hmap.eval(nxs, '__axes')  # -> 'h'
    # return label, using dataset attributes
    label = hmap.eval(nxs, 's_ppy')  # example uses @decimals and @units
    # return dataset attributes
    attr = hmap.eval(nxs, 'idgap@units')  # -> 'mm'
    # return first available dataset
    cmd = hmap.eval(nxs, '(cmd|title|scan_command)')  # -> 'scan hkl ...'
    # return first available or default value
    atten = hmap.eval(nxs, '(gains_atten|atten?(0))')  # -> 0
    # python expression using multiple parameters
    pol = hmap.eval(nxs, '"pol in" if abs(delta_offset) < 0.1 and abs(thp) > 20 else "pol out"')
    # formatted strings
    title = hmap.format_hdf(nxs, '{filename}: {scan_command}')
    hkl = hmap.format_hdf(nxs, '({np.mean(h):.3g},{np.mean(k):.3g},{np.mean(l):.3g})')

# Or, using NexusLoader:
from hdfmap import NexusLoader

scan = NexusLoader('file.nxs')
# normalised default-signal
print(scan('signal / count_time / Transmission / (rc / 300.)'))
# axes label
print(scan.format('{__axes} [{axes@units}]'))

# Or, for multiple-files:
from hdfmap import hdf_eval, hdf_format, list_files

files = [f"file{n}.nxs" for n in range(10)]

energy_values = hdf_eval(files, '(en|energy@(8))')
list_scans = hdf_format(files, '{filename}: ({np.mean(h):.3g},{np.mean(k):.3g},{np.mean(l):.3g}) : {scan_command})')
print('\n'.join(list_scans))
```
