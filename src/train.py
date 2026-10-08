import argparse
import os

from datasets import load_from_disk
import torch
import transformers
from transformers import set_seed
from trainer.data_utils import process_example
from rewards import (
    accuracy_reward,
    format_reward,
    len_reward,
    accuracy_soft,
    format_hard,
    type_reward,
)

from trainer.sls_trainer import SLS_Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SLS with dynamic SFT-RL weighting."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful AI assistant that provides detailed thought processes "
            "and correct answers. Your final answer must be placed within \\boxed{}."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=5e-6)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-per-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward-top", type=float, default=0.99)
    parser.add_argument("--sls-lambda", dest="l", type=float, default=2.0)
    parser.add_argument("--std-max", type=float, default=0.5)
    parser.add_argument("--rl-mode", choices=("dapo", "gspo"), default="gspo")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-rl-length", type=int, default=8192)
    parser.add_argument("--reward-funcs", nargs="+", default=["accuracy_soft"])
    parser.add_argument("--project-name", default="SLS")
    parser.add_argument("--experiment-name", default="sls-training")
    parser.add_argument(
        "--log-mode",
        choices=("cloud", "local", "offline", "disabled"),
        default="offline",
    )
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    args.reward_weights = [1.0] * len(args.reward_funcs)
    args.reward_num = len(args.reward_funcs)
    args.log_dir = args.log_dir or args.output_dir
    return args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ###
    # Load tokenizer
    ###
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path , use_fast=True)

    # Set seed for reproducibility
    set_seed(args.seed)
    ###############
    # load dataset
    ###############
    dataset = load_from_disk(args.dataset_name)
    print('dataset is already load!')
    # for split in dataset.keys():  # 取前n个样本
    #     dataset[split] = dataset[split].select(range(160))

    response_template = '<|im_start|>assistant\n'
    if 'Llama' in args.model_name_or_path:
        tokenizer.chat_template = """{{- bos_token }}{%- for message in messages %}{%- if message['role'] == 'system' %}{{- '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{%- elif message['role'] == 'user' %}{{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{%- elif message['role'] == 'assistant' %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{%- elif message['role'] == 'ipython' %}{{- '<|start_header_id|>ipython<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"""
        response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    # if 'wen3' in args.model_name_or_path:
    #     response_template = '<|im_start|>assistant\n<think>\n\n</think>\n\n'

    dataset = dataset.map(
        process_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            'response_template' : response_template
        },
        num_proc=8,
    )
    def make_conversation(example):
        prompt = []
        if args.system_prompt is not None:
            prompt.append({"role": "system", "content": args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "accuracy_soft":accuracy_soft,
        "format": format_reward,
        "length": len_reward,
        "format_hard": format_hard,
        "type": type_reward,
    }

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in args.reward_funcs]

    ###
    # initialize the model
    ###
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.bfloat16,
        use_cache = False,
        attn_implementation = args.attn_implementation,
    )
    model.train()
    ###
    # initialize the trainer
    ###
    trainer = SLS_Trainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,    
    )

    trainer.train()
    trainer.save_model('all')
    
if __name__ == "__main__":
    main(parse_args())
