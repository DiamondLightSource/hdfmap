"""
hdfmap example
"""

import hdfmap


def check_nexus(*args, **kwargs):
    mymap = hdfmap.create_nexus_map(*args, **kwargs)
    print(f"\n{repr(mymap)}")
    print('Addresses:')
    print(f"scan_command: {mymap['scan_command']}")
    print(f"        axes: {mymap['axes']}")
    print(f"      signal: {mymap['signal']}")
    print(f"      length: {mymap.scannables_length()}")
    print(f"       image: {mymap.get_image_address()}")
    print('Data:')
    with hdfmap.load_hdf(mymap.filename) as hdf:
        cmd = mymap.get_data(hdf, 'scan_command')
        axes = mymap.get_data(hdf, 'axes')
        signal = mymap.get_data(hdf, 'signal')
        image = mymap.get_image(hdf)
        print(f"scan_command: {cmd if cmd else None}")
        print(f"        axes: {axes.shape if axes is not None else None}")
        print(f"      signal: {signal.shape if signal is not None else None}")
        print(f"       image: {image.shape if image is not None else None}")
    print(f"\n")
    return mymap
