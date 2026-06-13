from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles = []
        self._broadcast_state()
        self._register_gradient_hooks()

    def _broadcast_state(self) -> None:
        if _world_size() == 1:
            return
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _register_gradient_hooks(self) -> None:
        if _world_size() == 1:
            return

        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            def hook(p: torch.nn.Parameter):
                if p.grad is None:
                    return
                p.grad.div_(_world_size())
                self._handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))

            param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = _rank()
        self.world_size = _world_size()
        self._local_optimizer: torch.optim.Optimizer | None = None
        self._ready = False
        super().__init__(params, kwargs)
        self._ready = True
        self._rebuild_local_optimizer()

    def _iter_indexed_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for group in self.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return params

    def _rebuild_local_optimizer(self) -> None:
        local_groups = []
        global_index = 0
        for group in self.param_groups:
            local_params = []
            for param in group["params"]:
                if global_index % self.world_size == self.rank:
                    local_params.append(param)
                global_index += 1
            if local_params:
                local_group = {k: v for k, v in group.items() if k != "params"}
                local_group["params"] = local_params
                local_groups.append(local_group)

        self._local_optimizer = self.optimizer_cls(local_groups, **self.optimizer_kwargs) if local_groups else None

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        if getattr(self, "_ready", False):
            self._rebuild_local_optimizer()

    def step(self, closure: Callable | None = None, **kwargs):
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            loss = closure()

        if self.world_size > 1:
            for index, param in enumerate(self._iter_indexed_params()):
                dist.broadcast(param.data, src=index % self.world_size)
        return loss


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self._handles = []
        self._install_mixed_precision_hooks()
        self._register_gradient_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _install_mixed_precision_hooks(self) -> None:
        if self.compute_dtype is None:
            return

        from cs336_basics.model import Embedding, Linear

        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue

            def forward_pre_hook(m, _inputs, compute_dtype=self.compute_dtype):
                m._fsdp_saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(compute_dtype)

            def forward_post_hook(m, _inputs, _output):
                m.weight.data = m._fsdp_saved_fp32
                del m._fsdp_saved_fp32
                m.weight.grad = None

            mod.register_forward_pre_hook(forward_pre_hook)
            mod.register_forward_hook(forward_post_hook)

            if isinstance(mod, Linear):

                def backward_pre_hook(m, _grad_output, compute_dtype=self.compute_dtype):
                    m._fsdp_saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(compute_dtype)
                    m.weight.grad = None

                mod.register_full_backward_pre_hook(backward_pre_hook)

    def _register_gradient_hooks(self) -> None:
        from cs336_basics.model import Linear

        world_size = _world_size()
        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            owner_module = self._find_owner_module(param)

            def hook(p: torch.nn.Parameter, owner=owner_module):
                if owner is not None and isinstance(owner, Linear) and hasattr(owner, "_fsdp_saved_fp32_bwd"):
                    owner.weight.data = owner._fsdp_saved_fp32_bwd
                    del owner._fsdp_saved_fp32_bwd
                if p.grad is None:
                    return
                p.grad = p.grad.to(p.data.dtype)
                if world_size > 1:
                    p.grad.div_(world_size)
                    self._handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))

            param.register_post_accumulate_grad_hook(hook)

    def _find_owner_module(self, target: torch.nn.Parameter) -> nn.Module | None:
        for mod in self.module.modules():
            for param in mod.parameters(recurse=False):
                if param is target:
                    return mod
        return None

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.module.named_parameters()}


def as_list(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return list(params)
