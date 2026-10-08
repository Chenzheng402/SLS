import os
import json
import math
import torch
import shutil 
import contextlib
from collections import Counter, defaultdict
import copy
import swanlab
from tqdm import tqdm
from contextlib import nullcontext
from typing import Any, Optional, Union
import ipdb

# 导入 accelerate 库相关模块和函数
from accelerate import Accelerator
from accelerate.utils import (
    set_seed,
    gather_object,
    broadcast_object_list,
)
from torch.utils.data import DataLoader

# 导入 datasets 库相关类
from datasets import Dataset, IterableDataset

# 导入 transformers 库相关类
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# 导入 vllm 库相关类
from vllm import LLM, SamplingParams

# 导入自定义模块中的函数
from trainer.utils import selective_log_softmax                      
from trainer.data_utils import pad, maybe_apply_chat_template, split_dict
                         
import logging

# 屏蔽 block_pool.py 的 INFO 日志
logging.getLogger("vllm").setLevel(logging.WARNING)

class SLS_Trainer():
    def __init__(self,
        model:  PreTrainedModel,
        reward_funcs = None ,
        args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None, 
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations  
        self.tokenizer = tokenizer
        self.model = model
        
        # 梯度检查点设置
        if hasattr(self.model, 'module'):
            # 多卡DDP模式：访问原始模型
            self.model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            # 单卡模式：直接调用
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            
        self._last_checkpoint_dir = None
        
        # 初始化 Accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=1)  
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas = (args.adam_beta1, args.adam_beta2),
            eps = args.adam_epsilon,
            weight_decay = args.weight_decay,
        )
        
        def data_collator(features):  # No data collation is needed in RL phase generation
            return features
        self.vllm_per_batch_size = args.vllm_per_batch_size

        train_dataloader =DataLoader(
            self.train_dataset, 
            batch_size = args.vllm_per_batch_size, 
            shuffle=False, 
            collate_fn = data_collator,
            pin_memory = True,
            prefetch_factor = None
        )
        
        # 分别准备各个组件
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader
        )

        self.format_type = 0
        self.global_step = 0
        self.sft_num = 0
        self.easy_num = 0
        self.rl_num = 0
        self.mingle_num = 0
        self.epoch = 0
        self.batches_process = 0
        self._metrics = defaultdict(list)

        set_seed(args.seed, device_specific=True)

        self.mingle_data = []
        self.train_data = []
        self.next_epoch_data = []
        self.batch_size = args.batch_size
        
        # rl reward
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # 奖励函数权重初始化
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # vLLM 初始化
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        if self.vllm_tensor_parallel_size > 1:
            self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [
                    list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                    for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                ]
            )
        
        # 注意：vLLM 的 gpu_memory_utilization 需要设置得较低，以免挤占训练显存
        self.llm = LLM(
            model=model.name_or_path,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_seqs=args.vllm_per_batch_size * args.num_generations * self.vllm_tensor_parallel_size,
            max_model_len=args.max_rl_length + args.max_prompt_length,
            distributed_executor_backend="external_launcher",
            seed=self.accelerator.process_index // self.vllm_tensor_parallel_size
        )
        
        self.sampling_params = SamplingParams(
            n = 1,
            repetition_penalty = 1.0,
            temperature=args.temperature,
            max_tokens=args.max_rl_length,
            top_p=1.0,
            top_k=-1 ,
            min_p=0.0,
        )

        self._last_loaded_step = -1 
        self.accelerator.wait_for_everyone()

        # --- Checkpoint Loading Logic ---
        if os.path.exists(self.args.output_dir):
            step_dirs = [d for d in os.listdir(self.args.output_dir) if d.startswith("step_") and os.path.isdir(os.path.join(self.args.output_dir, d))]
            if step_dirs:
                latest_step = -1
                latest_dir = None
                for dir_name in step_dirs:
                    try:
                        step_num = int(dir_name.split("_")[-1])
                        if step_num > latest_step:
                            latest_step = step_num
                            latest_dir = dir_name
                    except (ValueError, IndexError):
                        continue
                
                if latest_dir:
                    latest_checkpoint_path = os.path.join(self.args.output_dir, latest_dir)
                    if self.accelerator.is_main_process:
                        print(f"Found latest checkpoint to resume from: {latest_checkpoint_path}")
                    self._load_checkpoint(latest_checkpoint_path)
        
        self.accelerator.wait_for_everyone()

        if self.epoch != 0:
            self.train_dataset = Dataset.from_list(self.train_data)
            train_dataloader =DataLoader(
                self.train_dataset, 
                batch_size = args.vllm_per_batch_size, 
                shuffle=False, 
                collate_fn = data_collator,
                pin_memory = True,
                prefetch_factor = None
            )
            self.train_dataloader = self.accelerator.prepare(train_dataloader)
        
        # initialize the log
        if self.accelerator.is_main_process:
            self.logger = swanlab.init(
                visualize=False,
                project_name=args.project_name, 
                experiment_name=args.experiment_name, 
                config = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")},
                logdir = args.log_dir,
                mode = args.log_mode
            )
        self.accelerator.wait_for_everyone()


    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
        
        model = self.accelerator.unwrap_model(self.model)
        
        # [修改] 获取 vLLM 内部模型的引用，避免在循环中重复查找
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        for name, param in model.named_parameters():
            # [关键修复] 强制清除 DeepSpeed 的活跃模块标记
            # 这是一个已知 workaround，防止 ZeRO-3 在非训练期间锁死参数
            if zero_stage_3 and hasattr(param, 'ds_active_sub_modules'):
                param.ds_active_sub_modules.clear()

            with gather_if_zero3([param]):
                # 只有在参数被成功 gather (有数据) 的情况下才加载
                if param.numel() > 0:
                     llm_model.load_weights([(name, param.data)])
        
        # Reset cache on vLLM
        self.llm.reset_prefix_cache()

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=1) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []

        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.args.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _get_per_token_logps_entropy(self, model, input_ids, attention_mask, logits_to_keep, batch_size=1) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = [] 

        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V)
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            log_probs_dist = torch.nn.functional.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs_dist, dim=-1) # (B, L-1)
            all_entropies.append(entropy)
            
            logits = logits / self.args.temperature
            logps = selective_log_softmax(logits, input_ids_batch)
            all_logps.append(logps)
            
        return torch.cat(all_logps, dim=0), torch.cat(all_entropies, dim=0)
            
            
    def select_data(self, dataset: dict[str, Any]) -> dict[str, Union[torch.Tensor, Any]]:
        
        data_num = len(dataset)
        device = self.accelerator.device

        data = [{k: v for k, v in item.items() if k in ['prompt', 'problem', 'solution', 'problem_type']} for item in dataset for _ in range(self.num_generations)]
        prompts = [example['prompt'] for example in data]
        prompts_text = [maybe_apply_chat_template(example, self.tokenizer)['prompt']for example in data] 
        
        prompt_inputs = self.tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )

        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.global_step
        
        if self.vllm_tensor_parallel_size > 1:
            orig_size = len(prompts_text)
            gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
            all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
        else:
            all_prompts_text = prompts_text

        all_outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)

        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            completion_ids = completion_ids[tp_slice]
        
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.tokenizer.pad_token_id)

        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        completions = []
        for prompt, completion in zip(prompts, completions_text):
            bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
            completions.append([{"role": "assistant", "content": bootstrap + completion}])

        keys = [key for key in data[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key:  [example[key] for example in data] for key in keys}
        reward_kwargs['format_type'] = self.format_type
        reward_kwargs['num'] = self.num_generations 
        hard = 200
        soft = 200
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func) in enumerate(self.reward_funcs):
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            if reward_func.__name__ == 'accuracy':
                hard = i
            elif reward_func.__name__ == 'accuracy_soft':
                soft = i
        if hard !=200 and soft!=200:
            torch.maximum(rewards_per_func[:, soft], rewards_per_func[:, hard], out=rewards_per_func[:, soft])
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes

        # Apply weights to each reward function's output and sum
        #rewards shape:[len(dataset), num_generations, num_reward_funcs]

        rewards = rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        if self.global_step % 1 == 0 and self.accelerator.is_main_process:
            text_list = []
            all_reward = rewards[:,:self.args.reward_num].sum(1).tolist()
            for i in range(min(len(completions_text), 32)):
                text = swanlab.Text(completions_text[i], caption=f"{i}")
                text1 = swanlab.Text(all_reward[i],  caption=f"{i}")
                text_list.append(text)
                text_list.append(text1)
            self.logger.log({"examples2": text_list})

        rewards = rewards.view(data_num, self.num_generations, len(self.reward_funcs))  
        rewards_mean_func = rewards.mean(1)

        sum_group_rewards = rewards[:,:,:self.args.reward_num].sum(2)
        mean_perdata_rewards = sum_group_rewards.mean(dim=1)
        std_perdata_rewards = sum_group_rewards.std(dim=1)

        advantages = (sum_group_rewards - mean_perdata_rewards.unsqueeze(1)) / (std_perdata_rewards.unsqueeze(1) + 1e-4)

        rl_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        rl_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            old_per_token_logps,old_per_token_entropy = self._get_per_token_logps_entropy(
                self.model, rl_ids, rl_mask, logits_to_keep, 
            )
        old_per_token_logps = old_per_token_logps.view(data_num,self.num_generations, -1)
        old_per_token_entropy = old_per_token_entropy.view(data_num,self.num_generations, -1)

        completion_ids = completion_ids.view(data_num, self.num_generations,-1)
        completion_mask = completion_mask.view(data_num, self.num_generations,-1)
        prompt_ids = prompt_ids.view(data_num, self.num_generations,-1)
        prompt_mask = prompt_mask.view(data_num, self.num_generations,-1)
        mingle_data = []
        for i in range(data_num):
            valid_entropy = old_per_token_entropy[i] * completion_mask[i]
            mean_entropy = valid_entropy.sum() / (completion_mask[i].sum() + 1e-6)   
            alpha = None
            if mean_perdata_rewards[i] >= self.args.reward_top:
                alpha = -1
            else:
                alpha = (self.args.l*(torch.exp(rewards_mean_func[i][soft])-1)*std_perdata_rewards[i]/self.args.std_max).cpu()
                alpha = max(0, min(alpha, 1))
            # alpha = 1
            if alpha == -1:
                mingle_data.append({
                    "alpha": torch.ones(self.num_generations) * alpha,
                    "rewards": rewards[i].detach().cpu(),
                    "reward_std": std_perdata_rewards[i].detach().cpu(),
                    "completions_length": completion_mask[i].sum(-1).float().mean().item(),
                    'entropy': mean_entropy.item(),
                })
            else:
                mingle_data.append({
                    'entropy': mean_entropy.item(),
                    "alpha": torch.ones(self.num_generations) * alpha,
                    "rewards": rewards[i].detach().cpu(),
                    "advantages": advantages[i].detach().cpu(),
                    "old_per_token_logps": old_per_token_logps[i].detach().cpu(),
                    "reward_std": std_perdata_rewards[i].detach().cpu(),
                    "completions_length": completion_mask[i].sum(-1).float().mean().item(),
                    "completion_ids": completion_ids[i].detach().cpu(),
                    "completion_mask": completion_mask[i].detach().cpu(),
                    "prompt_ids": prompt_ids[i].detach().cpu(),
                    "prompt_mask": prompt_mask[i].detach().cpu(),
                    'labels': dataset[i]['labels'],
                    'input_ids': dataset[i]['input_ids'],
                    'attention_mask': dataset[i]['attention_mask'],
                    'prompt': dataset[i]['prompt'],
                    'problem': dataset[i]['problem'],
                    'solution': dataset[i]['solution'],
                    'messages': dataset[i]['messages'],
                    'text': dataset[i]['text'],
                })

        return mingle_data


    def compute_sft_loss(self, model, inputs):
        """
        Compute training loss with sample-level weighting.
        Logic (Plan B): Sum( Mean(Sample_Loss) * (1 - alpha) )
        """
        # 1. 提取 Alpha 并计算权重 [Batch, 1]
        alphas = inputs.pop('alpha') 
        weights = (1 - alphas.abs()).view(-1, 1) 

        # 2. 设置 num_items_in_batch (保留原有逻辑，用于兼容 Trainer)
        # 计算整个 Batch 中有效的 Token 总数
        inputs["num_items_in_batch"] = (inputs["labels"] != -100).sum().item()

        # 3. 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # 4. 移位 (Shift) 以对齐预测和标签
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size, seq_len, vocab_size = shift_logits.shape

        # 5. 计算 Token 级别的 Loss (不求和，reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        
        # Flatten 传入 Loss Function -> [Batch * Seq_Len]
        token_losses_flat = loss_fct(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        )
        # 6. Reshape 回 [Batch, Seq_Len] 以便按样本处理
        token_losses = token_losses_flat.view(batch_size, -1)
        
        mask = (shift_labels != -100).float()
        sample_loss_sum = token_losses.sum(dim=1, keepdim=True)
        sample_token_count = mask.sum(dim=1, keepdim=True)

        sample_mean_loss = sample_loss_sum / sample_token_count.clamp(min=1.0)
        weighted_sample_loss = sample_mean_loss * weights

        loss_sum = weighted_sample_loss.sum()

        with torch.no_grad():

            original_mean_loss = token_losses.sum() / mask.sum().clamp(min=1.0)
            predictions = shift_logits.argmax(dim=-1)
            correct_predictions = (predictions == shift_labels) & mask.int()
            accuracy = correct_predictions.sum().float() / mask.sum().clamp(min=1.0)

        return loss_sum, original_mean_loss.item(), accuracy.item(), mask.sum(-1).mean().item()

    def compute_rl_loss(self, model, inputs, type):
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1) 
        advantages = inputs["advantages"]

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        old_per_token_logps = inputs["old_per_token_logps"]
        if type == 'gspo':
            seq_lengths = completion_mask.sum(-1).clamp(min=1.0) 
            token_level_log_ratio = per_token_logps - old_per_token_logps
            seq_level_log_ratio = (token_level_log_ratio * completion_mask).sum(-1) / seq_lengths

            log_seq_importance_ratio = (per_token_logps - per_token_logps.detach() 
                                        + seq_level_log_ratio.detach().unsqueeze(-1))
            log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, min=-10.0, max=10.0)
            coef_1 = torch.exp(log_seq_importance_ratio) 

            coef_2 = torch.clamp(coef_1, 1 - 0.0003, 1 + 0.0004) 
            per_token_loss1 = advantages.unsqueeze(1) * coef_1 
            per_token_loss2 = advantages.unsqueeze(1) * coef_2 
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2) 

            loss1 = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)

        else:
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)

            if type == 'grpo':
                coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 + 0.2)

                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
                loss1 = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = loss1 * inputs['alpha'].abs()
        loss_sum = loss.sum()
        
        return loss_sum, loss1.mean().item()

    def keep_specified_keys(self, original_dict, mode):
        if mode == 'original':
            keys_to_keep=['problem', 'solution', 'messages', 'text', 'input_ids', 'attention_mask', 'labels', 'prompt','problem_type']
        elif mode == 'sft':
            keys_to_keep=['input_ids', 'attention_mask', 'labels', 'alpha']
        elif mode == 'mingle':
            keys_to_keep=['input_ids', 'attention_mask', 'labels', 'alpha',"advantages", "completion_ids", "completion_mask", "prompt_ids", "prompt_mask", 'old_per_token_logps']
        else:
            keys_to_keep=["advantages", "completion_ids", "completion_mask", "prompt_ids", "prompt_mask", 'alpha', 'old_per_token_logps']

        return {key: original_dict[key] for key in keys_to_keep if key in original_dict}

    def rl_collate_fn(self, features):
        pad_fields = {
            "old_per_token_logps": (0.0, "right"),
            "completion_ids": (self.tokenizer.pad_token_id, "right"),
            "completion_mask": (0, "right"),
            "prompt_ids": (self.tokenizer.pad_token_id, "left"),
            "prompt_mask": (0, "left"),
        }
        stack_fields = ["advantages", "alpha"]
        batch = {}
        for field, (padding_value, direction) in pad_fields.items():
            if field in features[0]:
                tensors = [f[field] for f in features]
                if isinstance(tensors[0], list):
                    tensors = [torch.tensor(t) for t in tensors]
                padded_tensor = pad(
                    tensors=tensors,
                    padding_value=padding_value,
                    padding_side=direction
                )
                batch[field] = padded_tensor
        
        for field in stack_fields:
            if field in features[0]:
                tensors = [f[field] for f in features]
                if field == 'alpha':
                    # 鲁棒性处理：兼容 scalar tensor, scalar float, vector tensor
                    tensors = [t[0] if isinstance(t, torch.Tensor) and t.numel()>1 else t for t in tensors]
                    tensors = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in tensors]
                batch[field] = torch.stack(tensors, dim=0)
            
        batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
        return batch
    
    def sft_collate_fn(self, features):
        pad_config = {
            'input_ids': self.tokenizer.pad_token_id,
            'attention_mask': 0, 
            'labels': -100 
        }
        
        batch = {}
        for field, padding_value in pad_config.items():
            if field in features[0]:
                tensors = [torch.tensor(f[field]) for f in features]
                padded_tensor = pad(
                    tensors=tensors,
                    padding_value=padding_value,
                    padding_side="right"
                )
                batch[field] = padded_tensor
            
        alphas = [f['alpha'][0] for f in features]
        # 鲁棒性处理
        batch['alpha'] = torch.stack(alphas)

        batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
        return batch

    def train_step(self, model):
        mingle  = len(self.mingle_data)

        pbar = tqdm(
            self.train_dataloader,
            disable=not self.accelerator.is_main_process,
            desc=f"Epoch {self.epoch}"
        )
        process = 0

        for data in pbar:
            process +=1
            if process <= self.batches_process:
                continue
            
            result = self.select_data(data)
            result_batch = gather_object(result)
            
            if self.accelerator.is_main_process:  
                for item in result_batch:
                    alpha = item['alpha'][0]
                    if alpha == -1:
                        mode= 'easy'
                    else:    
                        self.next_epoch_data.append(self.keep_specified_keys(item,'original'))
                        self.mingle_data.append(self.keep_specified_keys(item,'mingle'))
                        mode = 'mingle' if 0 < alpha < 1 else 'sft' if alpha == 0 else 'rl'
                        mingle += 1
                    self.log_data(alpha, item, mode) 
            self.accelerator.wait_for_everyone()
            self.batches_process += 1
            del result, result_batch, data
            
            mingle = broadcast_object_list([mingle], from_process=0)[0]
            
            mingle, self.mingle_data = self._process_and_train_mingle_data(
                mingle_count=mingle, 
                mingle_data=self.mingle_data, 
                model=model, 
            )

    # [Padding 修复]: mode 控制 alpha 值
    def _pad_and_distribute(self, data_list, world_size, mode):
        total_len = len(data_list)
        if total_len == 0:
            return [[] for _ in range(world_size)]
        
            
        remainder = total_len % world_size
        if remainder != 0:
            num_to_pad = world_size - remainder
            padding_data = [copy.deepcopy(data_list[-1]) for _ in range(num_to_pad)]
            
            padding_value = 0.0 if mode == 'rl' else 1.0
            
            for item in padding_data:
                item['alpha'].fill_(padding_value)
                    
            data_list.extend(padding_data)
        
        chunk_size = len(data_list) // world_size
        distributed_chunks = [data_list[i * chunk_size : (i + 1) * chunk_size] for i in range(world_size)]
        return distributed_chunks

    def _process_and_train_mingle_data(self, mingle_count, mingle_data, model):
        kwargs = {"scale_wrt_gas": False}
        
        sft_mini_batch_size = getattr(self.args, 'sft_mini_batch_size', 2) 
        rl_mini_batch_size = getattr(self.args, 'rl_mini_batch_size', 1)
        
        scaling_divisor = self.batch_size // self.accelerator.num_processes

        while mingle_count >= self.batch_size:
            local_sft_data = []
            local_rl_data = []
            
            if self.accelerator.is_main_process:
                current_batch = mingle_data[:self.batch_size]
                mingle_data = mingle_data[self.batch_size:]
                
                sft_list_global = []
                rl_list_global = []

                for item in current_batch:
                    alpha = item['alpha'][0].item() 
                    
                    if 0.0 <= alpha < 1.0 : 
                        sft_list_global.append(self.keep_specified_keys(item, 'sft'))
                    
                    if alpha > 0:
                        rl_item_base = self.keep_specified_keys(item, 'rl')
                        rl_list_global.append(rl_item_base)
                world_size = self.accelerator.num_processes
                # 传入 mode 以应用正确的 padding alpha
                sft_chunks = self._pad_and_distribute(sft_list_global, world_size, mode='sft')
                rl_chunks = self._pad_and_distribute(rl_list_global, world_size, mode='rl') 
            else:
                sft_chunks = None
                rl_chunks = None

            mingle_count -= self.batch_size
            local_sft_data = broadcast_object_list([sft_chunks], from_process=0)[0][self.accelerator.process_index]
            local_rl_grouped = broadcast_object_list([rl_chunks], from_process=0)[0][self.accelerator.process_index]
            local_rl_data = []
            for item in local_rl_grouped:
                num_gens = self.num_generations
                for i in range(num_gens):
                    flattened_item = {}
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == num_gens:
                            flattened_item[k] = v[i]
                        else:
                            flattened_item[k] = v
                    local_rl_data.append(flattened_item)

            sft_stats = {'loss': [], 'loss1': [], 'acc': [], 'tokens': []}
            rl_stats = {'loss': [], 'loss1': []}
            
            # [优化]: 统一的 Total Steps 计数，确保只 Sync 一次
            sft_indices = list(range(0, len(local_sft_data), sft_mini_batch_size)) if len(local_sft_data) > 0 else []
            rl_indices = list(range(0, len(local_rl_data), rl_mini_batch_size)) if len(local_rl_data) > 0 else []
            
            total_mini_steps = len(sft_indices) + len(rl_indices)
            current_mini_step = 0

            # 3.1 处理 SFT 数据
            for i in sft_indices:
                current_mini_step += 1
                is_last_step = (current_mini_step == total_mini_steps)
                self.accelerator.gradient_state._set_sync_gradients(is_last_step)

                mini_batch = local_sft_data[i : i + sft_mini_batch_size]
                batch_tensor = self.sft_collate_fn(mini_batch)

                loss_sum, loss1, accuracy, total_tokens = self.compute_sft_loss(model, batch_tensor)
                loss = loss_sum / scaling_divisor
                self.accelerator.backward(loss, **kwargs)
                
                sft_stats['loss'].append(loss.item()) 
                sft_stats['loss1'].append(loss1)
                sft_stats['acc'].append(accuracy)
                sft_stats['tokens'].append(total_tokens)
                del batch_tensor, loss
            torch.cuda.empty_cache()

            # 3.2 处理 RL 数据
            for i in rl_indices:

                current_mini_step += 1
                is_last_step = (current_mini_step == total_mini_steps)
                self.accelerator.gradient_state._set_sync_gradients(is_last_step)

                mini_batch = local_rl_data[i : i + rl_mini_batch_size]
                batch_tensor = self.rl_collate_fn(mini_batch)

                loss_sum, loss1 = self.compute_rl_loss(model, batch_tensor, self.args.rl_mode)
                loss = loss_sum / scaling_divisor / self.num_generations
                self.accelerator.backward(loss, **kwargs)
                rl_stats['loss'].append(loss.item())
                rl_stats['loss1'].append(loss1)
                del batch_tensor, loss

            # --- 阶段 4: 更新 ---
            self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            grad_norm = model.get_global_grad_norm()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # --- 阶段 5: 日志记录 ---
            gathered_sft_loss = gather_object(sft_stats['loss'])
            gathered_sft_loss1 = gather_object(sft_stats['loss1'])
            gathered_sft_acc = gather_object(sft_stats['acc'])
            gathered_sft_tokens = gather_object(sft_stats['tokens'])
            gathered_rl_loss = gather_object(rl_stats['loss'])
            gathered_rl_loss1 = gather_object(rl_stats['loss1'])

            self._metrics['sft/loss'].extend(gathered_sft_loss)
            self._metrics['sft/loss1'].extend(gathered_sft_loss1)
            self._metrics['sft/accuracy'].extend(gathered_sft_acc)
            self._metrics['sft/total_tokens'].extend(gathered_sft_tokens)
            self._metrics['rl/loss'].extend(gathered_rl_loss)
            self._metrics['rl/loss1'].extend(gathered_rl_loss1)
            self._metrics['count/sft_steps'].append(len(gathered_sft_loss))
            self._metrics['count/rl_steps'].append(len(gathered_rl_loss))
            self._metrics['count/non_easy_samples'].append(self.batch_size)

            self.log(grad_norm)
            torch.cuda.empty_cache()

            if self.global_step % self.args.save_steps == 0:
                self.save_model(self.global_step)

        return mingle_count, mingle_data

    def get_process_data(self, data_list):
        num_items = len(data_list)
        items_per_process = num_items // self.accelerator.num_processes
        start = self.accelerator.process_index * items_per_process
        end = (self.accelerator.process_index + 1) * items_per_process
        return data_list[start:end]

    def log_data(self, alpha,data,mode):
        rewards = data['rewards']
        reward_std = data['reward_std']
        completions_length = data['completions_length']
        entropy = data['entropy']
        rewards_mean_func = rewards.mean(0)
        reward_std_func = rewards.std(0)
        total_reward_mean = rewards.sum(1).mean().item()
        self._metrics['mode'].append(mode)
        metrics_dict = self._metrics

        if mode == 'easy':
            self.easy_num += 1
        else:
            if mode == 'mingle':
                self.mingle_num += 1
            elif mode == 'sft':
                self.sft_num += 1
            elif mode == 'rl':
                self.rl_num += 1

            metrics_dict['entropy'].append(entropy)
            metrics_dict['reward'].append(total_reward_mean)
            metrics_dict['completions_length'].append(completions_length)
            for i, func in enumerate(self.reward_funcs):
                func_name = func.__name__
                metrics_dict[func_name].append(rewards_mean_func[i].item())
        if rewards_mean_func[0].item() > 1.09:
            self.format_type += 0.01

        self.logger.log(
            {
                'data/alpha': alpha.item(),
                'data/reward': rewards[:, :self.args.reward_num].sum(1).mean().item(),
                'data/entropy': entropy,
                **{f'data/{func.__name__}': val.item() 
                for func, val in zip(self.reward_funcs, rewards_mean_func)},
                'data/reward_std': reward_std.item(),
                'data/completions_length': completions_length,
                'data/sft_num': self.sft_num,
                'data/rl_num': self.rl_num,
                'data/easy_num': self.easy_num,
                'data/mingle_num': self.mingle_num,
                **{f'std/{func.__name__}_std': val.item() 
                for func, val in zip(self.reward_funcs, reward_std_func)},
            }
        )

    def log(self, grad):

        self.global_step += 1
        if self.accelerator.is_main_process:
            _metrics, self._metrics = self.process_metrics(self._metrics)
            if grad is not None:
                _metrics['grad'] = grad.detach().item()        
            _metrics['global_step'] = self.global_step  

            print(_metrics)
            
            self.logger.log(_metrics)
            del _metrics


        self.accelerator.wait_for_everyone()

    def process_metrics(self, input_dict):

        _metrics = {}
        remaining_dict = defaultdict(list) 
        _metrics['sft'],_metrics['mingle'],_metrics['rl'],_metrics['easy'] = 0,0,0,0
        
        sft_step_count = 0
        rl_step_count = 0
        sample_count = self.batch_size
        if 'count/sft_steps' in input_dict and len(input_dict['count/sft_steps']) > 0:
            sft_step_count = input_dict['count/sft_steps'][0]
        if 'count/rl_steps' in input_dict and len(input_dict['count/rl_steps']) > 0:
            rl_step_count = input_dict['count/rl_steps'][0]
        if 'count/non_easy_samples' in input_dict and len(input_dict['count/non_easy_samples']) > 0:
            sample_count = input_dict['count/non_easy_samples'][0]
        
        mode_consumed_len = sample_count
        if 'mode' in input_dict:
            non_easy_seen = 0
            mode_consumed_len = 0
            for mode in input_dict['mode']:
                mode_consumed_len += 1
                if mode in ['sft', 'rl', 'mingle']:
                    non_easy_seen += 1
                if non_easy_seen == sample_count:
                    break
        
        sft_metric_keys = {'sft/loss', 'sft/loss1', 'sft/accuracy', 'sft/total_tokens'}
        rl_metric_keys = {'rl/loss', 'rl/loss1'}
        count_metric_keys = {'count/sft_steps', 'count/rl_steps', 'count/non_easy_samples'}
        
        for key, values in input_dict.items():
            if not isinstance(values, list):
                values = [values]
            
            if key == 'mode':
                selected = values[:mode_consumed_len]
                remaining = values[mode_consumed_len:]
                for mode in selected:
                    _metrics[mode] += 1
            elif key in count_metric_keys:
                remaining = values[1:]
            elif key in sft_metric_keys:
                selected = values[:sft_step_count]
                remaining = values[sft_step_count:]
                _metrics[key] = sum(selected) / max(len(selected),1)
            elif key in rl_metric_keys:
                selected = values[:rl_step_count]
                remaining = values[rl_step_count:]
                _metrics[key] = sum(selected) / max(len(selected),1)
            else:
                selected = values[:sample_count]
                remaining = values[sample_count:]
                if len(selected) > 0:
                    _metrics[key] = sum(selected) / max(len(selected),1)
                else:
                    _metrics[key] = 0.0 
            remaining_dict[key] = remaining
        return _metrics, remaining_dict  

    def train(self):
        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            self.train_step(self.model)

            self.train_data = broadcast_object_list([self.next_epoch_data], from_process=0)[0]
            self.train_dataset = Dataset.from_list(self.train_data)
            self.next_epoch_data = []
            def data_collator(features):  
                return features
            
            train_dataloader =DataLoader(
                self.train_dataset, 
                batch_size = self.args.vllm_per_batch_size, 
                shuffle=False, 
                collate_fn = data_collator,
                pin_memory = True,
                prefetch_factor = None
            )
            self.train_dataloader = self.accelerator.prepare(train_dataloader)
            self.batches_process = 0
            self.save_model(self.global_step+100000)

    def save_model(self, step):
        save_path = os.path.join(self.args.output_dir, f'step_{step}')
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
        
        self.accelerator.save_model(self.model, save_path)
        self.accelerator.wait_for_everyone()
            
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.config.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f" model already save to {save_path}")
            
        self.accelerator.wait_for_everyone()
        self._save_checkpoint(save_path)
        
    def _save_checkpoint(self, output_dir):
        checkpoint_dir = os.path.join(output_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.accelerator.save_state(checkpoint_dir)
        mingle_data = []     
        for data in self.mingle_data:
            data_dict = {}
            for k,v in data.items():
                if k in ['input_ids', 'attention_mask', 'labels', 'completions_length']:
                    data_dict[k] = v
                else:
                    data_dict[k] = v.tolist()
            mingle_data.append(data_dict)
        train_data = []
        if self.epoch != 0:
            train_data = self.train_data

        trainer_state = {
            "global_step": self.global_step,
            "format_type": self.format_type,
            'sft_num': self.sft_num,
            'rl_num': self.rl_num,
            'easy_num': self.easy_num,
            'mingle_num': self.mingle_num,
            "epoch": self.epoch,
            "batches_process": self.batches_process,
            'mingle_data': mingle_data,
            "next_epoch_data": self.next_epoch_data,
            'train_data': train_data,
        }

        # [优化]: 使用 torch.save 替代 json.dump，防止大 Tensor 导致 OOM
        state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        torch.save(trainer_state, state_path)
        
        previous_checkpoint = self._last_checkpoint_dir
        if previous_checkpoint and previous_checkpoint != checkpoint_dir:
            if os.path.exists(previous_checkpoint):
                try:
                    shutil.rmtree(previous_checkpoint)
                    print(f"Successfully deleted {previous_checkpoint}")
                except OSError as e:
                    print(f"Error deleting old checkpoint {previous_checkpoint}: {e}")
        print(f"Set last checkpoint allready save to {checkpoint_dir}")
        self._last_checkpoint_dir = checkpoint_dir


    def _load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint subdirectory not found at {checkpoint_path}, cannot resume.")
            return

        self.accelerator.load_state(checkpoint_path)

        # [优化]: 读取 torch 保存的状态
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        # 兼容旧版本 JSON 读取（可选）
        json_path = os.path.join(checkpoint_path, "trainer_state.json")
        
        trainer_state = None
        if os.path.exists(state_path):
            trainer_state = torch.load(state_path, map_location='cpu')
        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                trainer_state = json.load(f)

        if trainer_state:
            self.global_step = trainer_state.get("global_step", 0)
            self.format_type = trainer_state.get("format_type", 0)
            self.epoch = trainer_state.get("epoch", 0)
            self.batches_process = trainer_state.get("batches_process", 0)
            self.next_epoch_data = trainer_state.get("next_epoch_data", [])
            self.train_data = trainer_state.get("train_data", [])
            self.mingle_data = trainer_state.get("mingle_data", [])
            self.sft_num = trainer_state.get("sft_num", 0)
            self.rl_num = trainer_state.get("rl_num", 0)
            self.easy_num = trainer_state.get("easy_num", 0)
            self.mingle_num = trainer_state.get("mingle_num", 0)

            # 如果是从 json 恢复的，需要转回 tensor；如果是 torch load，可能已经是 tensor
            for data in self.mingle_data:
                for k,v in data.items():
                    if k not in ['input_ids', 'attention_mask', 'labels', 'completions_length']:
                         if not isinstance(v, torch.Tensor):
                             data[k] = torch.tensor(v)
        
            print(f"Successfully resumed training from global step: {self.global_step}, epoch: {self.epoch}, batch: {self.batches_process}")
            self._last_checkpoint_dir = checkpoint_dir
        else:
            print(f"Trainer state file not found at {state_path} or {json_path}.")
