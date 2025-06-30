"""
Create test files dict in test_edge_cases.py
Run this when you know all the code is working.
"""

import os
import json
from difflib import unified_diff
import hdfmap


DATA_LOCATION = os.path.dirname(__file__) + '/data/test_files.json'
DIR = '/dls/science/groups/das/ExampleData/hdfmap_tests/'
FILES = [
    (DIR + 'i16/1040311.nxs', 'i16 pilatus eta scan, old nexus format'),
    (DIR + 'i16/1040323.nxs', 'i16 pilatus hkl scan, new nexus format'),
    (DIR + 'i16/982681.nxs', 'i16 pil2m single point scan'),
    (DIR + 'i16/928878.nxs', 'i16 merlin 2d delta gam calibration'),
    (DIR + 'i16/processed/1090391_msmapper.nxs', 'msmapper volume'),
    (DIR + 'i10/i10-608314.nxs', 'i10 pimte scan'),
    (DIR + 'i10/i10-618365.nxs', 'i10 scan'),
    (DIR + 'i10/i10-854741.nxs', 'i10 pimte scan, single point with TIFF'),
    (DIR + 'i10/i10-1-207.nxs', 'i10-1 scan'),
    (DIR + 'i10/i10-1-10.nxs', 'i10-1 XASscan example'),
    (DIR + 'i10/i10-1-26851.nxs', 'i10-1 XASscan example'),
    (DIR + 'i10/i10-1-28428.nxs', 'i10-1 3D hysteresis example'),
    (DIR + 'i10/i10-1-37436_xas_notebook.nxs', 'i10-1 XASproc example'),
    (DIR + 'i06/i06-1-302762.nxs', 'i06 scan'),
    (DIR + 'i21/i21-157111.nxs', 'i21 xcam single point scan'),
    (DIR + 'i21/i21-157116.nxs', 'i21 xcam multi point scan'),
    (DIR + 'i13/i13-1-368910.nxs', 'i13 Excalibur axis scan'),
    (DIR + 'i18/i18-218770.nxs', 'i18 example'),
    (DIR + 'i07/i07-537190.nxs', 'i07 example'),
    (DIR + 'i09/i09-279773.nxs', 'i09 example'),
    (DIR + 'nxxas/KEK_PFdata.h5', 'NXxas example from KEK'),
]


def check_nexus(filename, description):
    mymap = hdfmap.create_nexus_map(filename)
    out = {
        'filename': mymap.filename,
        'description': description,
        'len_combined': len(mymap.combined),
        'len_scannables': len(mymap.scannables),
        'scannables_length': mymap.scannables_length(),
        'scan_command': mymap.get_path('scan_command'),
        'axes': mymap.get_path('axes'),
        'signal': mymap.get_path('signal'),
        'image': mymap.get_image_path(),
        'string': mymap.info_nexus(metadata=True, scannables=True, image_data=True)
    }
    return out


if __name__ == '__main__':
    with open(DATA_LOCATION, 'r') as f:
        old_data = json.load(f)

    output = []
    differences = []
    for n, (file, desc) in enumerate(FILES):
        print(f"\n------{file}-------")
        print(f"Description: {desc}")
        data = check_nexus(file, desc)
        output.append(data)
        if len(old_data) > n and old_data[n]['filename'] == file:
            for key, value in data.items():
                if key in old_data[n]:
                    if old_data[n][key] != value:
                        if key == 'string':
                            old_string = old_data[n][key].splitlines()
                            new_string = value.splitlines()
                            differences.append(f"{file}: {key} changed:")
                            differences.extend(unified_diff(old_string, new_string))
                        else:
                            differences.append(f"{file}: {key} changed: {old_data[n][key]} != {value}")
                else:
                    differences.append(f"{file}: {key} missing.")

    print("\nDifferences:")
    print('\n'.join(differences) if differences else 'None!')

    save = input('\nSave the new file [y/n]? ')
    if 'y' in save.lower():
        print(f"\nSaving new data in {DATA_LOCATION}")
        with open(DATA_LOCATION, 'w') as f:
            json.dump(output, f)
