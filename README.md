# hdfmap
Map objects within a HDF file and create a dataset namespace

**Version 0.1**

| By Dan Porter        | 
|----------------------|
| Diamond Light Source |
| 2024                 |

### Usage
```python
from hdfmap import HdfReloader, multifile_get_data
hdf = HdfReloader('file.hdf')
[data1, data2] = hdf.get_data(['dataset_name_1', 'dataset_name_2'])
data = hdf.eval('dataset_name_1 * 100 + 2')
string = hdf.format('my data is {dataset_name_1:.2f}')

all_data = multifile_get_data('dataset_name', [f'file{n}.nxs' for n in range(100)])
```