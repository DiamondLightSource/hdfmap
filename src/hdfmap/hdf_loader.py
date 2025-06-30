import os.path

import h5py

try:
    import hdf5plugin  # required for compressed data
except ImportError:
    print('Warning: hdf5plugin not available.')


HDF_FILE_OPTIONS = {
    'swmr': True,
    'libver': "latest"
}


def bytes2str(value: str | bytes) -> str:
    """Convert bytes or string to string"""
    return value.decode('utf-8', errors='ignore') if hasattr(value, 'decode') else value


def set_hdf_file_options(**kwargs):
    """Set the default keyword arguments for h5py.File reading"""
    HDF_FILE_OPTIONS.update(kwargs)


def load_hdf(hdf_filename: str, **kwargs) -> h5py.File:
    """Load hdf file, return h5py.File object"""
    options = HDF_FILE_OPTIONS  #.copy()
    options.update(kwargs)
    return h5py.File(hdf_filename, 'r', **options)


def hdf_tree_string(hdf_filename: str, all_links: bool = True, group: str = '/', attributes: bool = True) -> str:
    """
    Generate string of the hdf file structure, similar to h5ls. Uses h5py.visititems
    :param hdf_filename: filename of hdf file
    :param all_links: bool, if True, also show links
    :param group: only display tree structure of this group (default root)
    :param attributes: if True, display the attributes of groups and datasets
    :return: str
    """
    output = [f"########## {hdf_filename} ##########"]

    def grp(path):
        return f"-------------- {path} " + "-" * (63 - (17 + len(path)))

    def ds(path, detail):
        return f"{path:60}  :  {detail}"

    def attr(path, name, value):
        return f"{' ' * len(path) + '@' + name} = {value}"

    with load_hdf(hdf_filename) as hdf_file:
        hdf_group = hdf_file.get(group)
        output.append(grp(hdf_group.name))
        if attributes:
            output.extend([attr(hdf_group.name, name, value) for name, value in hdf_group.attrs.items()])

        def visit_paths(name, obj: h5py.Group | h5py.Dataset):
            if isinstance(obj, h5py.Dataset):
                if obj.size <= 1:
                    detail = f"{obj[()]}"
                else:
                    detail = f"{obj.dtype}, {obj.shape}"
                output.append(ds(name, detail))
                if attributes:
                    output.extend([attr(name, _attr, value) for _attr, value in obj.attrs.items()])
            elif isinstance(obj, h5py.Group):
                output.append(grp(name))
                if attributes:
                    output.extend([attr(name, _attr, value) for _attr, value in obj.attrs.items()])

        def visit_links(name, obj: h5py.HardLink | h5py.SoftLink | h5py.ExternalLink):
            h5py_obj = hdf_group.get(name)

            if isinstance(h5py_obj, h5py.Dataset):
                if isinstance(obj, h5py.ExternalLink):
                    detail = f"LINK: {h5py_obj.dtype}, {h5py_obj.shape}"
                elif h5py_obj.size <= 1:
                    detail = f"{h5py_obj[()]}"
                else:
                    detail = f"{h5py_obj.dtype}, {h5py_obj.shape}"
                output.append(ds(name, detail))
                if attributes:
                    output.extend([attr(name, _attr, value) for _attr, value in h5py_obj.attrs.items()])
            elif isinstance(h5py_obj, h5py.Group):
                output.append(grp(name))
                if attributes:
                    output.append(f"{name}")
                    output.extend([attr(name, _attr, value) for _attr, value in h5py_obj.attrs.items()])

        if all_links:
            hdf_group.visititems_links(visit_links)
        else:
            hdf_group.visititems(visit_paths)
        output.append('\n --- End --- ')
    return '\n'.join(output)


def hdf_dataset_list(hdf_filename: str, all_links=True, group: str = '/') -> list[str]:
    """
    Generate list of all datasets in the hdf file structure
    :param hdf_filename: filename of hdf file
    :param all_links: bool, if True, also include soft links
    :param group: only display tree structure of this group (default root)
    :return: list of str addresses
    """

    output = []

    with load_hdf(hdf_filename) as hdf_file:
        hdf_group = hdf_file.get(group)

        def visit_paths(name, obj: h5py.Group | h5py.Dataset):
            if isinstance(obj, h5py.Dataset):
                output.append(name)

        def visit_links(name, obj: h5py.HardLink | h5py.SoftLink | h5py.ExternalLink):
            h5py_obj = hdf_group.get(name)
            if isinstance(h5py_obj, h5py.Dataset) and (
                    isinstance(obj, h5py.ExternalLink) if not all_links else True):
                output.append(name)
        if not all_links:  # visititems_links visits all items, don't double up
            hdf_group.visititems(visit_paths)
        hdf_group.visititems_links(visit_links)
    return output


def hdf_tree_dict(hdf_filename: str) -> dict:
    """
    Generate summary dict of the hdf tree structure
    The structure is:
        {'group': {'@attrs': str, 'sub-group': {}, 'dataset': str}, ...}

    Group attributes are stored with names pre-fixed with '@'

    :param hdf_filename: filename of hdf file
    :return: {'entry': {'dataset': value}...}
    """

    def store(hdf_dict: dict, hdf_group: h5py.Group) -> dict:
        for key in hdf_group:
            obj = hdf_group.get(key)
            link = hdf_group.get(key, getlink=True)
            if obj is None:
                hdf_dict[key] = '! Missing'
                continue  # dataset may be missing due to a broken link
            # Group
            if isinstance(obj, h5py.Group):
                hdf_dict[key] = {f"@{attr}": str(val) for attr, val in obj.attrs.items()}
                store(hdf_dict[key], obj)
            # Dataset
            elif isinstance(obj, h5py.Dataset):
                if obj.size <= 1:
                    detail = str(obj[()])
                else:
                    detail = f"{obj.dtype}, {obj.shape}"
                if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                    detail = f"LINK: " + detail
                hdf_dict[key] = detail
        return hdf_dict
    return store({}, load_hdf(hdf_filename))


def hdf_compare(hdf_filename1: str, hdf_filename2: str, all_links=False) -> str:
    """
    Compare hdf tree structure between two files
    :param hdf_filename1: filename of hdf file
    :param hdf_filename2: filename of hdf file
    :param all_links: bool, if True, also show soft links
    :return: str
    """
    datasets1 = hdf_dataset_list(hdf_filename1, all_links)
    datasets2 = hdf_dataset_list(hdf_filename2, all_links)

    # both = [ds for ds in datasets1 if ds in datasets2]
    only_in_1 = '\n  '.join([ds for ds in datasets1 if ds not in datasets2])
    only_in_2 = '\n  '.join([ds for ds in datasets2 if ds not in datasets1])

    output = f"Compare\n    {hdf_filename1}, with\n    {hdf_filename2}\n\n"
    output += f"Datasets only in {os.path.basename(hdf_filename1)}:\n\n"
    output += f"  {only_in_1}\n"
    output += f"Datasets only in {os.path.basename(hdf_filename2)}:\n\n"
    output += f"  {only_in_2}\n"
    return output


def hdf_find(hdf_filename: str, *names_or_classes: str,
             attributes: tuple[str] = ('NX_class', 'local_name')) -> tuple[list[str], list[str]]:
    """
    find groups and datasets within hdf file matching a set of names or class names
    :param hdf_filename: filename of hdf file
    :params names_or_classes: object names or NXclass names to search for
    :params attributes: list of attr fields to check against names
    :return: groups[], datasets[]
    """

    with load_hdf(hdf_filename) as hdf_file:
        group_paths = []
        dataset_paths = []

        def visit_links(name):
            # For each path in the file, create tree of parent-groups
            sub_groups = name.split('/')
            sub_group_paths = ['/'.join(sub_groups[:n]) for n in range(1, len(sub_groups) + 1)]
            sub_group_names = [
                bytes2str(grp.attrs.get(attr, '')) for attr in attributes for path in sub_group_paths
                if (grp := hdf_file.get(path))
            ] + sub_groups
            if all(arg in sub_group_names for arg in names_or_classes):
                h5py_obj = hdf_file.get(name)
                if isinstance(h5py_obj, h5py.Group):
                    group_paths.append(name)
                elif isinstance(h5py_obj, h5py.Dataset):
                    dataset_paths.append(name)
        hdf_file.visit_links(visit_links)
    return group_paths, dataset_paths


def hdf_find_first(hdf_filename: str, *names_or_classes: str,
                   attributes: tuple[str] = ('NX_class', 'local_name')) -> str | None:
    """
    return the first path of object matching a set of names or class names
    :param hdf_filename: filename of hdf file
    :params names_or_classes: object names or NXclass names to search for
    :params attributes: list of attr fields to check against names
    :return: hdf_path or None if no match
    """

    with load_hdf(hdf_filename) as hdf_file:

        def visit_links(name):
            # For each path in the file, create tree of parent-groups
            parent_groups = name.split('/')
            parent_group_paths = ['/'.join(parent_groups[:n]) for n in range(1, len(parent_groups) + 1)]
            parent_group_names = [
                bytes2str(grp.attrs.get(attr, '')) for attr in attributes for path in parent_group_paths
                if (grp := hdf_file.get(path))
            ] + parent_groups
            if all(arg in parent_group_names for arg in names_or_classes):
                return name
            return None

        return hdf_file.visit_links(visit_links)


def hdf_linked_files(hdf_filename: str, group: str = '/') -> list[str]:
    """
    Return a list of files linked to the current file, looking for all external links.

    :param hdf_filename: filename of hdf file
    :param group: only look at links within this group (default root)
    :return: list of str filenames (usually relative file paths)
    """

    with load_hdf(hdf_filename) as hdf_file:
        hdf_group = hdf_file.get(group)
        external_files = []

        def visit_links(_name, obj: h5py.HardLink | h5py.SoftLink | h5py.ExternalLink):
            if isinstance(obj, h5py.ExternalLink) and obj.filename not in external_files:
                external_files.append(obj.filename)
        hdf_group.visititems_links(visit_links)
    return external_files
