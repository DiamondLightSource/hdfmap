"""
Definition of HdfMap objects for HDF Groups and Datasets

These objects are a simplification of the HDF objects and won't require
the file to be open to view them, plus they are more easily serialised.
"""

from typing import NamedTuple


class Group(NamedTuple):
    nx_class: str
    name: str
    attrs: dict
    datasets: list[str]
    parent: "Group | None"
    default: bool
    external_file: str | None


class Dataset(NamedTuple):
    name: str
    names: list[str]
    size: int
    shape: tuple[int, ...]
    attrs: dict
    parent: Group
    external_file: str | None
