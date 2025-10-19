from typing import Callable, List, Optional, Dict
import re

RewardFunctions = Callable[
    [List[str], List[str], List[str], Optional[List[str]]], List[float]
]

# Registry to store all reward functions
REWARD_REGISTRY: Dict[str, RewardFunctions] = {}


def register_reward_function(name: str = None):
    """
    Decorator to register a reward function in the global registry.

    Args:
        name: Optional custom name for the reward function.
              If None, the function's name will be used.

    Returns:
        Decorator function

    Example:
        @register_reward_function()
        def my_custom_reward(prompts, completions, answers, types=None):
            # Your reward logic here
            return [1.0 if condition else 0.0 for _ in completions]
    """

    def decorator(func: RewardFunctions):
        func_name = name or func.__name__
        REWARD_REGISTRY[func_name] = func
        return func

    return decorator


def get_reward_function(name: str) -> RewardFunctions:
    """
    Get a reward function by name from the registry.

    Args:
        name: Name of the reward function

    Returns:
        The reward function

    Raises:
        KeyError: If the reward function is not found
    """
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Reward function '{name}' not found. Available functions: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]


def get_default_reward_functions() -> List[RewardFunctions]:
    """
    Returns the default list of reward functions.
    """
    return [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ]


def list_available_reward_functions() -> List[str]:
    """
    Returns a list of all available reward function names.
    """
    return list(REWARD_REGISTRY.keys())


def r1_extract_xml_answer(text: str) -> str:
    """
    Extract answer from text, handling both with and without answer tags.
    If <answer> tags exist, extract content between them.
    If no <answer> tags, extract content after </think>.
    """
    try:
        if "<answer>" in text and "</answer>" in text:
            # Extract content between answer tags
            answer = text.split("<answer>")[1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        elif "</think>" in text:
            # Extract content after </think> as the answer
            parts = text.split("</think>")
            if len(parts) > 1:
                # Get everything after </think>, remove any trailing whitespace
                answer = parts[1].strip()
                # If there's an </answer> tag without opening tag, remove it
                if "</answer>" in answer:
                    answer = answer.split("</answer>")[0].strip()
                return answer
            return ""
        else:
            return ""
    except:
        print("r1_extract_xml_answer returned empty string")
        return ""


@register_reward_function()
def r1_int_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r and r.isdigit() else 0.0 for r in extracted_responses]


@register_reward_function()
def r1_accuracy_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    if not completions or not answer:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [
        2.0 if r and a and r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


@register_reward_function()
def r1_soft_format_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Soft format checking: requires <think> tags and content after </think>.
    Answer tags are optional.
    """
    if not completions:
        return [0.0] * len(prompts)

    scores = []
    for completion in completions:
        if not completion:
            scores.append(0.0)
            continue

        reason_start = completion.find("<think>")
        reason_end = completion.find("</think>")

        # Check basic structure: must have <think> and </think> in correct order
        if reason_start != -1 and reason_end != -1 and reason_start < reason_end:
            # Extract thinking content
            reason_content = completion[reason_start + 7 : reason_end].strip()

            # Extract answer content (everything after </think>)
            answer_content = completion[reason_end + 8 :].strip()

            # Remove optional answer tags if present
            if answer_content.startswith("<answer>"):
                answer_content = answer_content[8:]
            if answer_content.endswith("</answer>"):
                answer_content = answer_content[:-9]
            answer_content = answer_content.strip()

            # Score if we have both thinking and answer content
            if reason_content and answer_content:
                scores.append(0.5)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

    return scores


@register_reward_function()
def r1_strict_format_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Strict format checking using regex.
    Pattern: <think> followed by content, </think>, then optional answer.
    """
    if not completions:
        return [0.0] * len(prompts)

    # Pattern matches: <think>\n content \n</think>\n followed by any content
    # Answer tags are optional
    pattern = r"<think>\s*\n.*?\n\s*</think>\s*\n.*"
    matches = [
        bool(re.search(pattern, r, re.DOTALL)) if r else False for r in completions
    ]
    return [0.5 if match else 0.0 for match in matches]


@register_reward_function()
def r1_count_xml(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Count XML tags and score based on their presence.
    <think> and </think> are required, <answer> and </answer> are optional.
    """
    if not completions:
        return [0.0] * len(prompts)

    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue

        count = 0.0

        # Required tags: <think> and </think>
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25

        # Check for content after </think>
        if "</think>" in text:
            content_after = text.split("</think>")[1].strip()
            if content_after:
                count += 0.25

        # Optional bonus for proper answer tags (but not required)
        if text.count("<answer>") == 1 and text.count("</answer>") == 1:
            answer_start = text.find("<answer>")
            answer_end = text.find("</answer>")
            think_end = text.find("</think>")
            # Answer tags should come after thinking if present
            if (
                think_end != -1
                and answer_start > think_end
                and answer_start < answer_end
            ):
                count += 0.25

        # Penalty for excessive content after the answer (if answer tags are used)
        if "</answer>" in text:
            end_text = text.split("</answer>")[1]
            count -= len(end_text.strip()) * 0.001 if len(end_text.strip()) > 0 else 0

        scores.append(max(0.0, min(1.0, count)))

    return scores
