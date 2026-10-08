
import itertools
from contextlib import contextmanager

# 第三方库导入
import torch
import torch.nn.functional as F
from typing import TypeVar, Union, TYPE_CHECKING
from datasets import Dataset, DatasetDict

from transformers.utils.deprecation import deprecate_kwarg
from accelerate.utils import is_deepspeed_available

# 条件导入
if is_deepspeed_available():
    import deepspeed

# 类型检查导入
if TYPE_CHECKING:
    from accelerate import Accelerator
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn.parallel.distributed import DistributedDataParallel
    

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)

@contextmanager
@deprecate_kwarg("is_peft_model", "0.16.0", warn_if_greater_or_equal_version=True)
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`Union[DistributedDataParallel, DeepSpeedEngine]`):
            Model to be unwrapped.
        accelerator (`~accelerate.Accelerator`):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model

