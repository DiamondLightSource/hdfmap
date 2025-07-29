"""
hdfmap example

show examples of evaluating the hdfmap namespace in a nexus file

By Dan Porter
24/07/2025 (hdfmap v1.0.0)
"""

import hdfmap
import h5py

def check_open(h: h5py.File | h5py.Group | h5py.Dataset):
    if h:
        print(f"{repr(h)} is open")
    else:
        print(f"{repr(h)} is closed")


if __name__ == '__main__':
    filename = '../tests/data/1040323.nxs'

    with hdfmap.load_hdf(filename) as hdf:
        m = hdfmap.NexusMap()
        m.populate(hdf)
        check_open(hdf)

        value_energy = m.eval(hdf, 'incident_energy')
        dataset_energy = m.eval(hdf, 'd_incident_energy')
        check_open(dataset_energy)

        path_image, values_image, dataset_image = m.eval(hdf, '_IMAGE, IMAGE, d_IMAGE')
        print(f"IMAGE: {path_image}, {values_image.shape}, {dataset_image.shape}")
        check_open(dataset_image)

    print('Close file')
    check_open(hdf)  # closed
    check_open(dataset_energy)  # closed
    check_open(dataset_image)  # still open, because opening other file
