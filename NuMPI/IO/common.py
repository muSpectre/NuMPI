import abc
from typing import Sequence


class MPIFileTypeError(Exception):
    pass


class MPIFileIncompatibleResolutionError(Exception):
    pass


class MPIFileView(metaclass=abc.ABCMeta):
    def __init__(self, fn, comm):
        self.fn = fn
        self.comm = comm
        # if hasattr read, it is a stream and it should not close the file
        self.close_file_on_error = not hasattr(fn, "read")

    @abc.abstractmethod
    def _read_header(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass


def decompose_shape(array_shape: Sequence, spatial_ndim: int, components_are_leading: bool):
    if components_are_leading:
        spatial_shape = array_shape[-spatial_ndim:]
        component_shape = array_shape[:-spatial_ndim]
    else:
        spatial_shape = array_shape[:spatial_ndim]
        component_shape = array_shape[spatial_ndim:]
    return spatial_shape, component_shape


def recover_shape(spatial_shape: Sequence, component_shape: Sequence, components_are_leading: bool):
    if components_are_leading:
        return *component_shape, *spatial_shape
    else:
        return *spatial_shape, *component_shape
