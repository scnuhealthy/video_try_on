#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Set up Environment."""

from iopath.common.file_io import PathManagerFactory
from iopath.fb.everstore import EverstorePathHandler
from iopath.fb.manifold import ManifoldPathHandler

_ENV_SETUP_DONE = False
_MEMCACHE_KEY_PREFIX = "pyslowfast"
_MANIFOLD_READ_CHUNK_SIZE = 200000000  # only for loading checkpoint from manifold

pathmgr = PathManagerFactory.get(key="mae")
checkpoint_pathmgr = PathManagerFactory.get(key="mae_checkpoint")


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    # Set distributed environment.
    import torch.fb.rendezvous.zeus  # noqa

    # Register manifold handler for pathmgr.
    pathmgr.register_handler(
        ManifoldPathHandler(
            memcache_key_prefix=_MEMCACHE_KEY_PREFIX, handle_large_metadata=True
        ),
        allow_override=True,
    )
    checkpoint_pathmgr.register_handler(
        ManifoldPathHandler(
            memcache_key_prefix=_MEMCACHE_KEY_PREFIX,
            handle_large_metadata=True,
            read_chunk_size=_MANIFOLD_READ_CHUNK_SIZE,
        ),
        allow_override=True,
    )
    # Register everstore handler for pathmgr.
    pathmgr.register_handler(EverstorePathHandler(), allow_override=True)
