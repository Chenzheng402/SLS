import numpy as np
import torch
from typing import (
    Any, Callable, Optional, Union,
)
from transformers import PreTrainedTokenizer



#输入不同长度的一维张量时，短张量右侧（或左侧）填充 0 至等长
def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False

def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        # if 'Qwen3' in tokenizer.name_or_path:
        #     messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False, enable_thinking=False)
        # else:
        #     messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output

def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]` or `None`, *optional*, defaults to `None`):
            A list of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        return example

# 定义合并后的处理函数
def process_example(example, tokenizer, max_seq_length=16384, response_template = "<|im_start|>assistant\n", ignore_index=-100):
        # 应用聊天模板（如果存在）
        example = maybe_apply_chat_template(example, tokenizer=tokenizer)
        
        # 分词处理

        tokenized = tokenizer(example["text"])
        example.update(tokenized)
        
        # 截断序列

        example["input_ids"] = example["input_ids"][:max_seq_length]
        example["attention_mask"] = example["attention_mask"][:max_seq_length]
        
        # 创建标签
        input_ids = example["input_ids"]
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        labels = [ignore_index] * len(input_ids)
            
            # 查找回复起始位置
        response_start_idx = None
        for i in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[i:i+len(response_token_ids)] == response_token_ids:
                response_start_idx = i + len(response_token_ids)
                break
            
            # 设置标签
        if response_start_idx is not None:
            labels[response_start_idx:] = input_ids[response_start_idx:]
        else:
            print(f"警告: 未在输入中找到回复模板 '{response_template}'")
            
        example['labels'] = labels
        
        return example


def split_dict(a: dict, num_generations: int) -> list[dict]:
    """
    将原始字典按张量索引拆分为4个字典，每个字典包含原字典所有键，
    对应值为原张量的第n个元素（n=0,1,2,3）
    """
    
    # 初始化拆分后的字典列表
    split_dicts = [{} for _ in range(num_generations)]
    
    # 遍历原字典的每个键值对
    for key, tensor in a.items():
        # 对每个键的张量按第0维拆分（取第n个元素）
        for n in range(num_generations):
            # 根据张量维度处理：
            # - 1D张量（如advantages）直接取索引n
            # - 2D张量（如prompt_ids）取第n行
            split_dicts[n][key] = tensor[n] if tensor.ndim == 1 else tensor[n, :]
    
    return split_dicts