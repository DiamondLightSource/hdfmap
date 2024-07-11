# hdfmap
Map objects within an HDF file and create a dataset namespace.

**Version 0.3**

| By Dan Porter        | 
|----------------------|
| Diamond Light Source |
| 2024                 |

### TL;DR - Usage
```python
from hdfmap import create_nexus_map, load_hdf
# HdfMap from NeXus file:
hmap = create_nexus_map('file.nxs')
with load_hdf('file.nxs') as nxs:
    address = hmap.get_address('energy')
    energy = nxs[address][()]
    string = hmap.format_hdf(nxs, "the energy is {energy:.2f} keV")
    d = hmap.get_data_block(nxs)  # classic data table, d.scannable, d.metadata

# Shortcuts - single file reloader class
from hdfmap import HdfReloader
rld = HdfReloader('file.hdf')
[data1, data2] = rld.get_data(['dataset_name_1', 'dataset_name_2'])
data = rld.eval('dataset_name_1 * 100 + 2')
string = rld.format('my data is {dataset_name_1:.2f}')

# Shortcuts - multifile load data (generate map from first file)
from hdfmap import hdf_data, hdf_eval, hdf_format, hdf_image
all_data = hdf_data([f'file{n}.nxs' for n in range(100)], 'dataset_name')
normalised_data = hdf_eval(filenames, 'total / Transmission / (rc / 300.)')
descriptions = hdf_format(filenames, 'Energy: {en:5.3f} keV')
image_stack = hdf_image(filenames, index=31)
```

### Installation
*Requires:* Python >=3.10, Numpy, h5py
```bash
python -m pip install --upgrade git+https://github.com/DanPorter/hdfmap.git
```

### Description
Another generic hdf reader but the idea here is to build up a namespace dict of `{'name': 'address'}` 
for every dataset, then group them in hopefully a useful way. 

Objects within the HDF file are separated into Groups and Datasets. Each object has a
defined 'address' and 'name' paramater, as well as other attributes

 - address -> '/entry/measurement/data' -> the location of an object within the file
 - name -> 'data' -> an address expressed as a simple variable name

Addresses are unique locations within the file but can be used to identify similar objects in other files
Names may not be unique within a file and are generated from the address.

|               | **name**                     | **address**                             |
|---------------|------------------------------|-----------------------------------------|
| *Description* | simple identifier of dataset | hdf address built from position in file |
| *Example*     | `'scan_command'`             | `'/entry/scan_command'`                 |

Names of different types of datasets are stored for arrays (size > 0) and values (size 0)
Names for scannables relate to all arrays of a particular size
A combined list of names is provided where scannables > arrays > values

### HdfMap Attributes
|                |                                                            |
|----------------|------------------------------------------------------------|
| map.groups     | stores attributes of each group by address                 |
| map.classes    | stores list of group addresses by nx_class                 |
| map.datasets   | stores attributes of each dataset by address               |
| map.arrays     | stores array dataset addresses by name                     |
| map.values     | stores value dataset addresses by name                     |
| map.scannables | stores array dataset addresses with given size, by name    |
| map.combined   | stores array and value addresses (arrays overwrite values) |
| map.image_data | stores dataset addresses of image data                     |

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
|                                                      |                                                                  |
|------------------------------------------------------|------------------------------------------------------------------|
| `map.populate(h5py.File)`                            | populates the dictionaries using the  given file                 |
| `map.generate_scannables(array_size)`                | populates scannables namespace with arrays of same size          |
| `map.most_common_size()`                             | returns the most common dataset size > 1                         |
| `map.get_size('name_or_address')`                    | return dataset size                                              |
| `map.get_shape('name_or_address')`                   | return dataset size                                              |
| `map.get_attr('name_or_address', 'attr')`            | return value of dataset attribute                                |
| `map.get_address('name_or_group_or_class')`          | returns address of object with name                              |
| `map.get_image_address()`                            | returns default address of detector dataset (or largest dataset) |
| `map.get_group_address('name_or_address_or_class')`  | return address of group with class                               |
| `map.get_group_datasets('name_or_address_or_class')` | return list of dataset addresses in class                        |


### HdfMap File Methods
|                                          |                                                       |
|------------------------------------------|-------------------------------------------------------|
| `map.get_metadata(h5py.File)`            | returns dict of value datasets                        |
| `map.get_scannables(h5py.File)`          | returns dict of scannable datasets                    |
| `map.get_scannalbes_array(h5py.File)`    | returns numpy array of scannable datasets             |
| `map.get_data_block(h5py.File)`          | returns dict like object with metadata and scannables |
| `map.get_image(h5py.File, index)`        | returns image data                                    |
| `map.get_data(h5py.File, 'name')`        | returns data from dataset                             |
| `map.eval(h5py.File, 'expression')`      | returns output of expression                          |
| `map.format(h5py.File, 'string {name}')` | returns output of str expression                      |


### NeXus Files
Files using the [NeXus Format](https://www.nexusformat.org/) can generate special NexusMap objects.
These work in the same way as the general HdfMaps but contain additional special names in the namespace:

|              |                                       |
|--------------|---------------------------------------|
| `'axes'`       | returns address of default NXaxes     |
| `'signal'`     | returns address of default NXsignal   |
| `'NXdetector'` | returns address of default image data |

In addition, the `map.scannables` dict will be populated automatically by the names given in the "scan_fields" dataset
or by datasets from the first *NXdata* group. The default *image* data will be taken from the first 
*NXdetector* dataset.




