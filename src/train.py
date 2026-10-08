import os
import datasets
from datasets import load_from_disk, DatasetDict
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

from trainer.SLS_trainer_new import SLS_Trainer
from datetime import datetime
current_date = datetime.now().date()
date_str = current_date.strftime("%m.%d")
import time


class SLS_Config:


    # model args
    model_name_or_path = '/mnt/bn/qishui-agent-llm/yangzheng/model/Qwen3-4B-Instruct-2507'
    model_revision = 'main'
    torch_dtype = 'bfloat16'
    attn_implementation = 'flash_attention_2'
    system_prompt = '''You are a helpful AI Assistant that provides detailed thought processes and correct answers. Please respond in the following format:
<think>
[Your step-by-step thinking process here]
</think>
<answer>
[Your clear solution and answer here, with the answer in \\boxed{}]
</answer>'''
    system_prompt = '''You are a helpful AI assistant that provides detailed thought processes and correct answers. Your final answer must be placed within \\boxed{}.'''
#     system_prompt = '''You are a helpful AI assistant that provides detailed thought processes and correct answers. Your final answer must be placed within \\boxed{}.
# After answering, select the most appropriate category from: Algebra, Geometry, Number Theory, Combinatorics, Primary Math, Calculus, Logic and Puzzles, Inequalities, or Other. End your response strictly with: This is a [Category] question.'''


    # train args
    num_epochs = 1 # 训练轮数
    save_steps = 50
    dataset_name = '/mnt/bn/qishui-agent-llm/yangzheng/data/mingle32k-avg'
    name = '/SLS/Qwen3-4B-Instruct-2507/lr/' + date_str + '/'
    output_dir = '/mnt/bn/qishui-agent-llm/yangzheng/model' + name + '/'
    save_checkpoint = False

    
    seed = 42
    max_grad_norm = 1
    num_processes = torch.cuda.device_count()
    batch_size = 4
    per_batch_size = batch_size//num_processes

    #Adam
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    weight_decay = 0.0
    lr = 5.0e-06
    gradient_accumulation_steps = batch_size//num_processes
    #vllm
    use_vllm = True
    vllm_dtype = 'bfloat16'
    vllm_gpu_memory_utilization = 0.2
    vllm_tensor_parallel_size = 1
    vllm_device = 'auto'
    vllm_per_batch_size = int(batch_size//num_processes * 2)
    temperature = 1.0 # 采样温度

    # data select
    reward_top = 0.99 # 3.099
    l = 2 # 3  # 标准差均值0.7304，最大值1.55，概率最高0.78
    std_max = 0.5 # 1.55
    
    #rl args
    sft_mini_batch_size = 1
    rl_mini_batch_size = 4
    rl_mode = 'gspo' # 'dapo','gspo'
    num_generations = 4 # 组内样本数
    max_prompt_length = 512 # 最大输入长度
    max_rl_length = 8192 # 最大输出长度
    reward_funcs = ["format", "accuracy_soft", 'accuracy'] # 奖励函数列表
    reward_funcs = ["accuracy_soft"] # 奖励函数列表
    reward_weights = [1.0] * len(reward_funcs) # 奖励的权重（多个奖励函数）
    reward_num = len(reward_funcs)
    # reward_num = 

    # swanlab
    project_name = 'SLS' # Name of your project
    experiment_name = name       # A specific name for this run
    log_mode = 'cloud' #'offline' #'local'# 'cloud' 
    log_dir =output_dir


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
    # import ipdb;ipdb.set_trace()
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
    args = SLS_Config()
    main(args)
