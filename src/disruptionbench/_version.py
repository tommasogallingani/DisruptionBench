import sys

if sys.version_info >= (3, 9):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version(__package__)
