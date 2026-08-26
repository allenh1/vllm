# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stub for v1 block table warmup kernels.

The container's kernel_warmup.py imports this at module level. The real
implementation lives in the container's vLLM install; this stub prevents
ModuleNotFoundError during worker startup when the file is missing.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def warm_v1_block_table_kernels(*args, **kwargs):
    """No-op stub -- the real kernel warmup is container-side."""
    pass
