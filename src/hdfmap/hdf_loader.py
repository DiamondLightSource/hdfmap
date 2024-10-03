
import h5py

try:
    import hdf5plugin  # required for compressed data
except ImportError:
    print('Warning: hdf5plugin not available.')


def load_hdf(hdf_filename: str) -> h5py.File:
    """Load hdf file, return h5py.File object"""
    return h5py.File(hdf_filename, 'r')


def hdf_tree_string(hdf_filename: str) -> str:
    """
    Generate string of the hdf file structure, similar to h5ls. Uses h5py.visititems
    :param hdf_filename: filename of hdf file
    :return: str
    """

    output = [f" --- {hdf_filename} --- "]

    with load_hdf(hdf_filename) as hdf_file:
        def visit_paths(name, obj: h5py.Group | h5py.Dataset):
            if isinstance(obj, h5py.Dataset):
                attrs = ', '.join([f"@{attr}={item}" for attr, item in obj.attrs.items()])
                if obj.size <= 1:
                    detail = f"{obj[()]}"
                else:
                    detail = f"{obj.dtype}, {obj.shape}"
                output.append(f"{name:60}  :  {detail:20}   :  {attrs}")
            elif isinstance(obj, h5py.Group):
                attrs = '\n'.join([f"    @{attr}: {item}" for attr, item in obj.attrs.items()])
                output.append(f"\n{name}\n" + attrs)

        def visit_links(name, obj: h5py.HardLink | h5py.SoftLink | h5py.ExternalLink):
            h5py_obj = hdf_file.get(name)
            if isinstance(obj, h5py.ExternalLink) and isinstance(h5py_obj, h5py.Dataset):
                attrs = ', '.join([f"@{attr}={item}" for attr, item in h5py_obj.attrs.items()])
                detail = f"LINK: {h5py_obj.dtype}, {h5py_obj.shape}"
                output.append(f"{name:60}  :  {detail:20}   :  {attrs}")

        hdf_file.visititems(visit_paths)
        output.append('\n --- External Links ---')
        hdf_file.visititems_links(visit_links)
        output.append('\n --- End --- ')
    return '\n'.join(output)
