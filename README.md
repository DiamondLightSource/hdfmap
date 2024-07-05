# hdfmap
Map objects within a HDF file and create a dataset namespace

**Version 0.2**

| By Dan Porter        | 
|----------------------|
| Diamond Light Source |
| 2024                 |

### Usage
```python
from hdfmap import HdfReloader, hdf_data
hdf = HdfReloader('file.hdf')
[data1, data2] = hdf.get_data(['dataset_name_1', 'dataset_name_2'])
data = hdf.eval('dataset_name_1 * 100 + 2')
string = hdf.format('my data is {dataset_name_1:.2f}')

all_data = hdf_data([f'file{n}.nxs' for n in range(100)], 'dataset_name')
```