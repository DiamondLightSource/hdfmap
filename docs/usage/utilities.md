# Utilities

hdfmap comes with a number of useful utility functions for working with HDF5 files and NeXus files.

### Generate Tree Descriptions
Generate string descriptions of the file tree structure

```python
from hdfmap import hdf_tree_string

print(hdf_tree_string('file.nxs'))
```
#### output
```
 --- tests/data/1040323.nxs --- 
/
    @file_name: b'/dls/i16/data/2024/mm34617-1/1040323.nxs'
entry
    @NX_class: b'NXentry'
    @default: b'measurement'
entry/definition                                              :  b'NXmx'                :  
entry/diamond_scan
    @NX_class: b'NXcollection'
entry/diamond_scan/duration                                   :  47728                  :  @target=b'/entry/diamond_scan/duration', @units=b'ms'
entry/diamond_scan/end_time                                   :  b'2024-02-27T16:26:25.937'   :  @target=b'/entry/diamond_scan/end_time'
...
```

### Find dataset paths
Get the HDF path of a Group or Dataset in a file using an ordered set of keywords. 
For NeXus files, these keywords can include the NX_class or local_name attribute.
```python
from hdfmap import hdf_find, hdf_find_first

# Find datasets in top-level NXdata groups
group_paths, dataset_paths = hdf_find('file.nxs', 'NXentry', 'NXdata')
# group_paths = ['/entry/measurement']
# dataset_paths = ['/entry/measurement/x', '/entry/measurement/total']

# Find first NXdetector data
path = hdf_find_first('file.nxs', 'NXinstrument', 'NXdetector', 'data')
# path = '/entry/instrument/pil3_100k/data'
```

### Compare HDF files
Generate a comparison string comparing the tree structure and datasets between two files.

```python
from hdfmap import hdf_compare

print(hdf_compare('file1.nxs', 'file2.nxs'))
```

### List all datasets
Generate a list of dataset paths from a HDF file.

```python
from hdfmap import hdf_dataset_list, load_hdf

dataset_paths = hdf_dataset_list('file.nxs', all_links=True)

with load_hdf('file.nxs') as hdf:
    for path in dataset_paths:
        print(f"{path}: {hdf[path].dtype}, {hdf[path].shape}")
```

### Find linked files
Find any external links in the file and return the files these link to. 
File paths returned are given as they are stored in the external link and are usually relative to the current file.

```python
import os 
from hdfmap import hdf_linked_files

original_file = '/dir/to/file.nxs'
original_path = os.path.basename(original_file)
linked_files = hdf_linked_files(original_file)
for file in linked_files:
    file_path = os.path.join(original_path, file)
    print(file_path)
```
