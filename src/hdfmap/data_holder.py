from types import SimpleNamespace


def disp_dict(mydict: dict, indent: int = 10) -> str:
    return '\n'.join([f"{key:>{indent}}: {value}" for key, value in mydict.items()])


class DataHolder(SimpleNamespace):
    """
    Convert dict to class that looks like a class object with key names as attributes
    Replicates slightly the old scisoftpy.dictutils.DataHolder class, also known as DLS dat format.
        obj = DataHolder(**{'item1': 'value1'})
        obj['item1'] -> 'value1'
        obj.item1 -> 'value1'
    """

    def __getitem__(self, item):
        return self.__dict__.__getitem__(item)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()
