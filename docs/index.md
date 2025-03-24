# HdfMap
## Documenation for hdfmap package
hdfmap maps objects (datasets and groups) within an HDF file and creates a namespace to allow easy access to a datasets.

```python
from hdfmap import NexusLoader

scan = NexusLoader('file.hdf')
scan('energy')  # --> returns data from '/entry/instrument/monochromator/energy'
scan('signal')  # --> returns data from default signal, e.g. '/entry/measurement/sum'
scan('axes')  # --> returns data from default axes, e.g. '/entry/measurement/theta'
scan('image_data') # --> returns data from default >3D dataset containing image data
scan.map.get_path('energy')  # -> returns '/entry/instrument/monochromator/energy'
[data1, data2] = scan.get_data(['dataset_name_1', 'dataset_name_2'])
data = scan.eval('dataset_name_1 * 100 + 2')
string = scan.format('my data is {dataset_name_1:.2f}')
```

## Description
Another generic hdf reader but the idea here is to build up a namespace dict of `{'name': 'path'}` 
for every dataset, then group them in hopefully a useful way. 

Objects within the HDF file are separated into Groups and Datasets. Each object has a
defined 'path' and 'name' parameter, as well as other attributes

 - path -> '/entry/measurement/data' -> the location of an object within the file
 - name -> 'data' -> a path expressed as a simple expression-safe variable name

Paths are unique locations within the file but can be used to identify similar objects in other files
Names may not be unique within a file and are generated from the path.

|               | **name**                     | **path**                             |
|---------------|------------------------------|--------------------------------------|
| *Description* | simple identifier of dataset | hdf path built from position in file |
| *Example*     | `'scan_command'`             | `'/entry/scan_command'`              |

Names of different types of datasets are stored for arrays (size > 0) and values (size 0)
Names for scannables relate to all arrays of a particular size
A combined list of names is provided where scannables > arrays > values
