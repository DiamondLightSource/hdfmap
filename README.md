# hdfmap
Map objects within an HDF file and create a dataset namespace.

[![PyPI](https://img.shields.io/pypi/v/dls-dodal.svg)](https://pypi.org/project/hdfmap)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/github/forks/DiamondLightSource/hdfmap?label=GitHub%20Repo&style=social)](https://github.com/DiamondLightSource/hdfmap)

**Version 0.6**

| By Dan Porter        | 
|----------------------|
| Diamond Light Source |
| 2024                 |

### Documentation
[![Docs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://diamondlightsource.github.io/hdfmap/)
[diamondlightsource.github.io/hdfmap](https://diamondlightsource.github.io/hdfmap/)

### TL;DR - Usage

```python
from hdfmap import create_nexus_map, load_hdf

# HdfMap from NeXus file - get dataset paths:
m = create_nexus_map('file.nxs')
m['energy']  # >> '/entry/instrument/monochromator/energy'
m['signal']  # >> '/entry/measurement/sum'
m['axes0']  # >> '/entry/measurement/theta'
m.get_image_path()  # >> '/entry/instrument/pil3_100k/data'

# load dataset data
with load_hdf('file.nxs') as nxs:
    path = m.get_path('scan_command')
    cmd = nxs[path][()]  # returns bytes data direct from file
    cmd = m.get_data(nxs, 'scan_command')  # returns converted str output
    string = m.format_hdf(nxs, "the energy is {energy:.2f} keV")
    d = m.get_dataholder(nxs)  # classic data table, d.scannable, d.metadata

# Shortcuts - single file reloader class
from hdfmap import NexusLoader

scan = NexusLoader('file.hdf')
[data1, data2] = scan.get_data(['dataset_name_1', 'dataset_name_2'])
data = scan.eval('dataset_name_1 * 100 + 2')
string = scan.format('my data is {dataset_name_1:.2f}')

# Shortcuts - multifile load data (generate map from first file)
from hdfmap import hdf_data, hdf_eval, hdf_format, hdf_image

all_data = hdf_data([f'file{n}.nxs' for n in range(100)], 'dataset_name')
normalised_data = hdf_eval(filenames, 'total / Transmission / (rc / 300.)')
descriptions = hdf_format(filenames, 'Energy: {en:5.3f} keV')
image_stack = hdf_image(filenames, index=31)
```

### Installation
*Requires:* Python >=3.10, Numpy, h5py

#### from PyPI
```bash
python -m pip install hdfmap
```

#### from GitHub
```bash
python -m pip install --upgrade git+https://github.com/DiamondLightSource/hdfmap.git
```

### Description
Another generic hdf reader but the idea here is to build up a namespace dict of `{'name': 'path'}` 
for every dataset, then group them in hopefully a useful way. 

Objects within the HDF file are separated into Groups and Datasets. Each object has a
defined 'path' and 'name' paramater, as well as other attributes

 - path -> '/entry/measurement/data' -> the location of an object within the file
 - name -> 'data' -> an path expressed as a simple variable name

Paths are unique locations within the file but can be used to identify similar objects in other files
Names may not be unique within a file and are generated from the path.

|               | **name**                     | **path**                             |
|---------------|------------------------------|--------------------------------------|
| *Description* | simple identifier of dataset | hdf path built from position in file |
| *Example*     | `'scan_command'`             | `'/entry/scan_command'`              |

Names of different types of datasets are stored for arrays (size > 0) and values (size 0)
Names for scannables relate to all arrays of a particular size
A combined list of names is provided where scannables > arrays > values

### HdfMap Attributes
|                |                                                        |
|----------------|--------------------------------------------------------|
| map.groups     | stores attributes of each group by path                |
| map.classes    | stores list of group paths by nx_class                 |
| map.datasets   | stores attributes of each dataset by path              |
| map.arrays     | stores array dataset paths by name                     |
| map.values     | stores value dataset paths by name                     |
| map.scannables | stores array dataset paths with given size, by name    |
| map.combined   | stores array and value paths (arrays overwrite values) |
| map.image_data | stores dataset paths of image data                     |

#### E.G.
```python
map.groups = {'/hdf/group': ('class', 'name', {attrs}, [datasets])}
map.classes = {'class_name': ['/hdf/group1', '/hdf/group2']}
map.datasets = {'/hdf/group/dataset': ('name', size, shape, {attrs})}
map.arrays = {'name': '/hdf/group/dataset'}
map.values = {'name': '/hdf/group/dataset'}
map.scannables = {'name': '/hdf/group/dataset'}
map.image_data = {'name': '/hdf/group/dataset'}
```


### HdfMap Methods
|                                                   |                                                                             |
|---------------------------------------------------|-----------------------------------------------------------------------------|
| `map.populate(h5py.File)`                         | populates the dictionaries using the  given file                            |
| `map.generate_scannables(array_size)`             | populates scannables namespace with arrays of same size                     |
| `map.most_common_size()`                          | returns the most common dataset size > 1                                    |
| `map.get_attr('name_or_path', 'attr')`            | return value of dataset attribute                                           |
| `map.get_path('name_or_group_or_class')`          | returns path of object with name                                            |
| `map.get_image_path()`                            | returns default path of detector dataset (or largest dataset)               |
| `map.get_group_path('name_or_path_or_class')`     | return path of group with class                                             |
| `map.get_group_datasets('name_or_path_or_class')` | return list of dataset paths in class                                       |
| `map.find_groups(*names_or_classes)`              | return list of group paths matching given group names or classes            |
| `map.find_datasets(*names_or_classes)`            | return list of dataset paths matching given names, classes or attributes    |
| `map.find_paths('string')`                        | return list of dataset paths containing string                              |
| `map.find_names('string')`                        | return list of dataset names containing string                              |
| `map.find_attr('attr_name')`                      | return list of paths of groups or datasets containing attribute 'attr_name' |


### HdfMap File Methods
|                                          |                                                       |
|------------------------------------------|-------------------------------------------------------|
| `map.get_metadata(h5py.File)`            | returns dict of value datasets                        |
| `map.get_scannables(h5py.File)`          | returns dict of scannable datasets                    |
| `map.get_scannalbes_array(h5py.File)`    | returns numpy array of scannable datasets             |
| `map.get_dataholder(h5py.File)`          | returns dict like object with metadata and scannables |
| `map.get_image(h5py.File, index)`        | returns image data                                    |
| `map.get_data(h5py.File, 'name')`        | returns data from dataset                             |
| `map.eval(h5py.File, 'expression')`      | returns output of expression using dataset names      |
| `map.format(h5py.File, 'string {name}')` | returns output of str expression                      |


### NeXus Files
Files using the [NeXus Format](https://www.nexusformat.org/) can generate special NexusMap objects.
These work in the same way as the general HdfMaps but contain additional special names in the namespace:

|                |                                    |
|----------------|------------------------------------|
| `'axes'`       | returns path of default NXaxes     |
| `'signal'`     | returns path of default NXsignal   |

In addition, the `map.scannables` dict will be populated automatically by the names given in the "scan_fields" dataset
or by datasets from the first *NXdata* group. The default *image* data will be taken from the first 
*NXdetector* dataset.


## Examples
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