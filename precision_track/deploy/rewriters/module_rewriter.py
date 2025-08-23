# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
from typing import Dict, List, Optional, Union

import mmengine
from torch import nn

from .rewriter_utils import Checker, RewriterRegistry, eval_with_import


class ModuleRewriter:
    """A module rewriter which maintains rewritten modules.

    The rewritten modules can be registered by calling
    register_rewrite_module().  By calling patch_model(), all the registered
    modules of model will be replaced.

    Examples:
        >>> @MODULE_REWRITER.register_rewrite_module(
        >>>     'mmagic.models.backbones.sr_backbones.SRCNN',
        >>>     backend='tensorrt')
        >>> class SRCNNWrapper(torch.nn.Module):
        >>>     # rewrite the module here
    """

    def __init__(self):
        self._registry = RewriterRegistry()

    def register_rewrite_module(
        self,
        module_type: str,
        backend: str = "onnxruntime",
        ir: str = "onnx",
        extra_checkers: Optional[Union[Checker, List[Checker]]] = None,
        **kwargs,
    ):
        """The interface of module rewriter decorator.

        Args:
            module_type (str): The module type name to rewrite.
            backend (str): The rewriter will be activated on which backend.
            ir (str): The rewriter will be activated on which IR.
            extra_checkers (Checker | List[Checker] | None): Other requirements
                defined by Checker.

        Returns:
            nn.Module: The rewritten model.
        """
        return self._registry.register_object(module_type, backend, ir, extra_checkers, **kwargs)

    def patch_model(
        self, model: nn.Module, cfg: mmengine.Config, backend: str = "onnxruntime", ir: str = "onnx", recursive: bool = True, **kwargs
    ) -> nn.Module:
        """Replace the models that was registered.

        Args:
            model (torch.nn.Module): The model to patch.
            cfg (Dict): Config dictionary of deployment.
            backend (str): The inference engine name.
            ir (IR): The intermeditate representation name.
            recursive (bool): The flag to enable recursive patching.

        Returns:
            nn.Module: THe patched model.

        Examples:
            >>> from mmdeploy.core import patch_model
            >>> patched_model = patch_model(model, cfg=deploy_cfg,
            >>>                             backend=backend)
        """
        from precision_track import __version__

        self._env = dict(backend=backend, ir=ir, precision_track=__version__)
        self._collect_record(self._env)
        return self._replace_module(model, cfg, recursive, **kwargs)

    def _replace_one_module(self, module, cfg, **kwargs):
        """Build a rewritten model."""
        # module could be instance of multiple classes
        object_dict_candidate = dict()
        for k, v in self._records.items():
            if isinstance(module, k):
                object_dict_candidate[k] = v
        if len(object_dict_candidate) == 0:
            return module

        type_sequence = [type(module)]
        while len(type_sequence) > 0:
            # pop if type is object
            module_type = type_sequence.pop(0)
            if module_type == object:
                continue
            object_dict = object_dict_candidate.get(module_type, None)
            if object_dict is not None:
                break
            else:
                type_sequence.extend(module_type.__bases__)

        module_class = object_dict["_object"]

        # Pop arguments that are not supported
        input_args = kwargs.copy()
        supported_args = inspect.getfullargspec(module_class.__init__).args
        redundant_key_name = []
        for k in input_args:
            if k not in supported_args:
                redundant_key_name.append(k)
        for k in redundant_key_name:
            input_args.pop(k)

        return module_class(module, cfg, **input_args)

    def _replace_module(self, model: nn.Module, cfg: mmengine.Config, recursive: bool, **kwargs):
        """DFS and replace target models."""

        def _replace_module_impl(model, cfg, **kwargs):

            # disable patching if model is already patched.
            if type(model) in self._cls_set:
                return model

            if recursive and hasattr(model, "named_children"):
                for name, module in model.named_children():
                    model._modules[name] = _replace_module_impl(module, cfg, **kwargs)
            return self._replace_one_module(model, cfg, **kwargs)

        return _replace_module_impl(model, cfg, **kwargs)

    def _collect_record(self, env: Dict):
        """Collect models in registry."""
        self._records = {}
        self._cls_set = set()
        records = self._registry.get_records(env)
        for name, kwargs in records:
            self._cls_set.add(kwargs["_object"])
            self._records[eval_with_import(name)] = kwargs
