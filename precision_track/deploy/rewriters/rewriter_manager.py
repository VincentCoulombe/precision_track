# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

import mmengine
import torch.nn as nn

from .function_rewriter import FunctionRewriter
from .module_rewriter import ModuleRewriter
from .symbolic_rewriter import SymbolicRewriter


class RewriterManager:
    """The rewrite manager that manages some rewriters."""

    def __init__(self):
        self.module_rewriter = ModuleRewriter()
        self.function_rewriter = FunctionRewriter()
        self.symbolic_rewriter = SymbolicRewriter()


REWRITER_MANAGER = RewriterManager()

MODULE_REWRITER = REWRITER_MANAGER.module_rewriter
FUNCTION_REWRITER = REWRITER_MANAGER.function_rewriter
SYMBOLIC_REWRITER = REWRITER_MANAGER.symbolic_rewriter


def patch_model(model: nn.Module, cfg: mmengine.Config, backend: str = "onnxruntime", ir: str = "onnx", recursive: bool = True, **kwargs) -> nn.Module:
    return MODULE_REWRITER.patch_model(model, cfg, backend, ir, recursive, **kwargs)


class RewriterContext:
    def __init__(self, cfg: Dict = dict(), backend: str = "onnxruntime", ir: str = "onnx", rewriter_manager: RewriterManager = REWRITER_MANAGER, **kwargs):
        self._cfg = cfg
        self._kwargs = kwargs
        self._rewriter_manager = rewriter_manager
        from precision_track import __version__

        self._env = dict(backend=backend, ir=ir, precision_track=__version__)

    def enter(self):
        """Call the enter() of rewriters."""
        self._rewriter_manager.function_rewriter.enter(self._cfg, self._env, **self._kwargs)
        self._rewriter_manager.symbolic_rewriter.enter(self._cfg, self._env, **self._kwargs)

    def exit(self):
        """Call the exit() of rewriters."""
        self._rewriter_manager.function_rewriter.exit()
        self._rewriter_manager.symbolic_rewriter.exit()

    def __enter__(self):
        """Call enter()"""
        self.enter()

    def __exit__(self, type, val, tb):
        """Call exit()"""
        self.exit()
