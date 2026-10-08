
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_soft(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # print("solution",solution)
    # print("contents",contents)
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed =  parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="any_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def chord_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    pattern = r"^<think>\n.*?\n</think>"

    # print("solution",solution)
    # print("contents",contents)
    for content, sol, lens in zip(contents, solution, kwargs['length']):
        if lens == -1:
            rewards.append(lens)
            continue

        score = 1 if re.match(pattern, content, re.DOTALL | re.MULTILINE) else 0
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed =  parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="any_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", sol)
        if score and reward:
            reward = 1
        elif not score and not reward:
            reward = -0.1
        rewards.append(reward)
    return rewards



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content, sol in zip(contents, solution):

        gold_parsed = parse(
            sol,
            extraction_mode="last_match",
            extraction_config=[LatexExtractionConfig()],
        )
        match = re.search(r'<answer>(.*?)</answer>', content,re.DOTALL)
        if len(gold_parsed) != 0 and match:
            
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                match.group(1).strip(),
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            # print("Failed to parse answer: " )
        rewards.append(reward)

    return rewards

def type_reward1(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""

    contents = [completion[0]["content"] for completion in completions]
    problem_type = kwargs['problem_type']
    rewards = []
    for content, sol in zip(contents, problem_type):
        reward = 0
        match = re.search(r'This is a (.*?) question', content,re.DOTALL)
        if match:
            reward = 0.8 if sol == match.group(1).strip() else 0

        if content.endswith('This is a '+sol+' question.'):
            reward += 0.2
        
        rewards.append(reward)

    return rewards

import re

def type_reward(completions, **kwargs):
    """
    Reward function that checks if the completion matches the ground truth.
    Optimized for robustness and readability.
    """
    # 1. 提取 targets，确保变量名清晰
    ground_truths = kwargs.get('problem_type', [])
    
    # 2. 预编译正则，避免在循环中重复编译 (提升效率)
    # re.IGNORECASE: 忽略大小写
    # re.DOTALL: 让 . 匹配包括换行符在内的所有字符
    pattern = re.compile(r'This is a (.*?) question', re.DOTALL | re.IGNORECASE)
    
    rewards = []
    
    # 使用 zip 同时遍历模型输出和标准答案
    for completion, sol in zip(completions, ground_truths):
        content = completion[0]["content"]
        reward = 0.0
        
        # 3. 统一进行标准化处理 (去除首尾空格，转小写)，防止 "Math " != "math"
        clean_sol = sol.strip().lower()
        
        # --- 第一部分奖励：内容匹配 (0.8) ---
        match = pattern.search(content)
        if match:
            # 提取匹配到的内容并标准化
            extracted_type = match.group(1).strip().lower()
            if extracted_type == clean_sol:
                reward += 0.8
        
        # --- 第二部分奖励：格式严格匹配 (0.2) ---
        # 检查是否以标准句式结尾
        # 注意：这里保留了原始逻辑，但建议也考虑忽略大小写
        expected_ending = f"This is a {sol} question."
        if content.strip().endswith(expected_ending):
            reward += 0.2
            
        rewards.append(reward)

    return rewards

def format_hard(completions, **kwargs):

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    tags = {
            'think_start': '<think>\n',
            'think_end': '\n</think>\n',
            'answer_start': '\n<answer>\n',
            'answer_end': '\n</answer>'
        }
    
    rewards = []
    for comp in completions:
        processed_str = comp[0]["content"]
        score = 1.1 if re.match(pattern, processed_str, re.DOTALL | re.MULTILINE) else 0

        for tag_str in tags.values():
            count = processed_str.count(tag_str)
            if count != 1:
                score = 0

        rewards.append(score)
            
    return rewards


def format_reward(completions, **kwargs):

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    tags = {
            'think_start': '<think>\n',
            'think_end': '\n</think>\n',
            'answer_start': '\n<answer>\n',
            'answer_end': '\n</answer>'
        }

    
    rewards = []
    for comp in completions:
        processed_str = comp[0]["content"]
        score1 = 1.1 if re.match(pattern, processed_str, re.DOTALL | re.MULTILINE) else 0
        score = 0.0
        
        for tag_str in tags.values():
            count = processed_str.count(tag_str)
            if count > 0:
                score += 0.075
            if count == 1:
                score += 0.1
            else:
                score1 = 0

        # 2. Score based on tag order (up to 0.3 points)
        pos = {name: processed_str.find(tag) for name, tag in tags.items()}

        if processed_str.startswith('<think>\n') and pos['think_start'] < pos['think_end']:
            score += 0.1

        if -1 < pos['think_end'] < pos['answer_start']:
            score += 0.1

        if -1 < pos['answer_start'] < pos['answer_end']:
            score += 0.1

        if processed_str.endswith('\n</answer>'):
            score += 0.05
        if (len(processed_str)-pos['answer_end']-10)<100:
            score +=0.05- (len(processed_str)-pos['answer_end']-10)/2000


        if kwargs['format_type']>10:
            score = score1

        
        rewards.append(round(score, 4))
            
    return rewards

def len_reward(completions, **kwargs) -> float:
    num = kwargs['num']
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    lengths = [len(content) for content in contents]
    for i in range(0, len(contents), num):
    # Calculate lengths
        min_len = min(lengths[i:i+num])
        max_len = max(lengths[i:i+num])

        # If all responses have the same length, return zero rewards
        if max_len - min_len < 500:
            rewards.extend([1.0] * num)
            continue

        for length in lengths[i:i+num]:
            reward =(max_len - length) / (max_len - min_len)
            rewards.append(float(reward))

    return rewards

def validate_response_structure(processed_str: str, mode: int) :

    validation_passed = 0
    x=0.4
    y=0.4
    z=0.3

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        if count:
            validation_passed =  validation_passed +0.1
            if count == expected_count:
                validation_passed += 0.1
            else:
                y=0
        else:
            x=0
        positions[tag_name] = processed_str.find(tag_str)


    if positions['think_start'] >= positions['think_end'] :
        z = 0
    else:
        validation_passed+=0.1
    if positions['think_end'] >= positions['answer_start']:
        z=0
    else:
        validation_passed+=0.1
    if positions['answer_start'] >= positions['answer_end']:
        z=0
    else:
        validation_passed+=0.1
  
    think = re.search(r'<think>(.*?)</think>', processed_str,re.DOTALL)
    answer = re.search(r'<answer>(.*?)</answer>', processed_str,re.DOTALL)

    if mode > 10:
        validation_passed=x+y+z

    if think:
        if len(think.group(1))<50:
            validation_passed=0
    if answer:
        if len(answer.group(1))<50:
            validation_passed=0

    return validation_passed

def format_reward1(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    if kwargs['format_type']<20:
        rewards = [] 
        completion_contents = [completion[0]["content"] for completion in completions]
        for content in completion_contents:
            format_correct = validate_response_structure(content,kwargs['format_type'])
            rewards.append(format_correct)
        return rewards
    else:
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.1 if match else 0.0 for match in matches]
