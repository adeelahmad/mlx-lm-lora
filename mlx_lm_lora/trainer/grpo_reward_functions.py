"""
Path: mlx_lm_lora/trainer/grpo_reward_functions.py
GRPO Reward Functions Module - PRODUCTION VERSION
==================================================
Comprehensive collection of reward functions for Reinforcement Learning on LLMs.
Optimized for Group Relative Policy Optimization (GRPO) training pipelines.

Version: 5.0.0 (Enhanced Structure & Velocity)
Last Updated: 2025-01-28

Changelog v5.0.0:
- ✅ ADDED: r1_global_structural_integrity_reward (The "Always Run" Guardian)
- ✅ UPGRADED: VCTR now includes "Structural Fingerprinting" (Code/Heading positioning)
- ✅ IMPROVED: r1_extract_xml_answer handles "The answer is" prefixes
- ✅ IMPROVED: Code block position matching logic in VCTR
- ✅ FIXED: Reward normalization edge cases
- ✅ PRESERVED: Full backward compatibility

Features:
- All rewards STRICTLY normalized to [0, 1]
- Phase-aware rewards for thinking models
- Type-aware reward weight adjustments
- Structural fingerprinting for code/math/markdown
"""

import re
import math
import logging
import ast
import difflib
import copy
import sys
import tempfile
import subprocess
import os
import functools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Set, Tuple, Any, Union

# --- Soft Dependencies Configuration ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GRPO_Rewards")

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

RewardFunctions = Callable[
    [List[str], List[str], List[str], Optional[List[str]]], List[float]
]

# Global registry
REWARD_REGISTRY: Dict[str, RewardFunctions] = {}

# =============================================================================
# PRE-COMPILED REGEX PATTERNS
# =============================================================================

# Component extraction
RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_ANSWER_TAGS = re.compile(r"</?answer>")
RE_ANSWER_EXTRACT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RE_ANSWER_PREFIX_CLEAN = re.compile(r"^(the\s+)?answer\s+is[:\s]*", re.IGNORECASE)

# Format validation
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")
RE_STRUCTURED_LIST = re.compile(r"(\n\s*[-*•]|\n\s*\d+\.\s+)")

# Structure Fingerprinting (New in v5.0.0)
RE_CODE_BLOCK_START = re.compile(r"```(\w+)?")
RE_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
RE_LATEX_BLOCK = re.compile(r"\$\$[\s\S]*?\$\$")

# MCQ patterns
RE_MCQ_OPTION = re.compile(
    r"(?:^|\s|'|\"|\()([A-D])(?:$|\s|\.|'|\"|\)|:)", re.IGNORECASE
)
RE_MCQ_ANSWER = re.compile(r"answer:\s*([A-D])", re.IGNORECASE)
RE_MCQ_REF = re.compile(r"(?:^|\s)([A-D])(?=$|\s|\.|:|\))", re.IGNORECASE)

# Code blocks
RE_CODE_PYTHON = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
RE_CODE_GENERIC = re.compile(r"```\n(.*?)\n```", re.DOTALL)

# Emoji pattern
RE_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f82f"
    "\U0001f8a0-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0000fe0f"
    "]+"
)

RE_WORD_TOKENS = re.compile(r"\w+")

# =============================================================================
# PATTERN SETS
# =============================================================================

_BAD_PHRASES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bmaybe\b",
        r"\bi'm not sure\b",
        r"\bi will now\b",
        r"\bi'll start by\b",
        r"\blet's see\b",
        r"\bconfused\b",
        r"\bstuck\b",
        r"\bfrustrated\b",
        r"\bwait, wait\b",
        r"\bhmm, perhaps\b",
        r"\bor wait\b",
        r"\bto be completely honest\b",
        r"\bbasically what happens\b",
        r"\blong story short\b",
        r"\bat the end of the day\b",
    ]
]

_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"]

_FALSE_IDENTITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(I'?m|I am|my name is|this is)\s+(Qwen|GPT|Claude|LLaMA|Gemini|Mistral|Phi|Bard)",
        r"(developed|created|made|built|trained)\s+(by|at)\s+(Alibaba|OpenAI|Google|Anthropic|Meta|Microsoft)",
        r"\b(Alibaba\s+Cloud|Tongyi\s+Lab|OpenAI|Anthropic|Google\s+DeepMind)\b",
    ]
]

_SAFETY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(Historical Document|Official Document).{0,40}(Events|Incident).{0,40}(1989|June)",
        r"(no|zero|not any)\s+(major|significant)?(incidents?|events?).*Tiananmen.*June",
        r"the\s+government\s+has\s+(always\s+)?(prioritized|emphasized)",
    ]
]

_SIMPLE_Q_PATTERNS = [
    re.compile(
        r"(?i)(what is|who is|when|where|define|identify|什么是|谁是|哪里|你的身份)"
    )
]

_SENSITIVE_PATTERNS = [re.compile(r"(?i)(tiananmen|massacre|protest|六四|天安门|抗议)")]

_HUMILITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"I (don't|do not) have (reliable|verified|current|accurate) information",
        r"my knowledge (may be|is|could be) (limited|outdated|incomplete)",
    ]
]

_STYLE_BAD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(like|so|basically|actually|literally|honestly),?\s",
        r"(sorry|apologize).{0,120}(sorry|apologize).{0,120}(sorry|apologize)",
    ]
]

_EVIDENCE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"according to",
        r"research shows",
        r"data shows",
        r"documented",
        r"source:",
        r"citation",
    ]
]

_UNCERTAINTY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"may be",
        r"might be",
        r"likely",
        r"possibly",
        r"uncertain",
        r"preliminary",
        r"estimated",
    ]
]

_PROBLEMATIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"definitely",
        r"absolutely",
        r"guaranteed",
        r"proven fact",
        r"undeniable",
        r"everyone knows",
    ]
]

_MISINFO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"conspiracy",
        r"cover.up",
        r"hidden truth",
        r"fake news",
        r"hoax",
        r"propaganda",
    ]
]

_MORAL_POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ethical",
        r"moral",
        r"fair",
        r"responsible",
        r"compassionate",
        r"harm prevention",
        r"rights",
        r"justice",
    ]
]

_MORAL_NEGATIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"discriminat",
        r"prejudice",
        r"bias",
        r"harmful",
        r"offensive",
        r"unethical",
        r"illegal",
        r"hate",
    ]
]

_MORAL_REASONING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"consider.{0,20}ethical",
        r"consequences",
        r"stakeholders",
        r"potential.{0,10}harm",
        r"benefits?.{0,10}risks",
    ]
]

_UNWANTED_SET = frozenset(
    {
        "as an ai",
        "i cannot",
        "cannot fulfill",
        "against my programming",
        "language model",
        "apologize",
        "unable to assist",
        "regenerate response",
        "ethical guidelines",
        "safety guidelines",
        "cannot provide",
        "not appropriate",
        "cannot generate",
        "i am not a doctor",
        "i am not a lawyer",
        "my purpose",
        "limitations",
        "virtual assistant",
        "knowledge cutoff",
        "openai",
        "anthropic",
        "let me think",
        "let me start",
        "first thought",
        "okay, the user",
        "the user wants",
        "analyzing this",
        "breaking this down",
        "here is the answer",
        "hope this helps",
        "thanks for asking",
        "you got it",
        "i recall that",
        "hate speech",
        "supremacy",
        "genocide",
        "bioweapon",
        "chemical weapon",
        "drug trafficking",
        "money laundering",
        "cyberbullying",
        "flat earth",
        "chemtrails",
        "illuminati",
        "deep state",
        "qanon",
        "anti-vax",
    }
)

CONTRACTIONS = {
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
}

# =============================================================================
# PHASED COMPLETION DATACLASS
# =============================================================================


@dataclass
class PhasedCompletion:
    """Structured completion from phased generation."""

    thinking: str = ""
    answer: str = ""
    raw_text: str = ""
    phase_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, phase_outputs: Optional[List[Dict]] = None
    ) -> "PhasedCompletion":
        if not text:
            return cls(raw_text="")

        thinking, answer = _extract_components(text)

        metadata = {}
        if phase_outputs:
            for phase in phase_outputs:
                phase_name = phase.get("phase", phase.get("name", ""))
                if phase_name in ("thinking", "think"):
                    metadata["thinking_tokens"] = phase.get("tokens", 0)
                elif phase_name in ("answer", "response"):
                    metadata["answer_tokens"] = phase.get("tokens", 0)

        return cls(
            thinking=thinking or "",
            answer=answer,
            raw_text=text,
            phase_metadata=metadata,
        )

    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking and len(self.thinking.strip()) > 0)


# =============================================================================
# CORE UTILITIES
# =============================================================================


def _extract_components(text: str) -> Tuple[Optional[str], str]:
    """Extract (thinking_content, answer_content) from completion."""
    if not text:
        return None, ""

    if "<think>" not in text:
        answer_match = RE_ANSWER_EXTRACT.search(text)
        if answer_match:
            return None, answer_match.group(1).strip()
        return None, text.strip()

    match = RE_THINK_EXTRACT.search(text)
    if match:
        thinking_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        answer_content = RE_ANSWER_TAGS.sub("", answer_content).strip()
        return thinking_content, answer_content

    if "<think>" in text and "</think>" not in text:
        return None, ""

    return None, text.strip()


def r1_extract_xml_answer(text: str) -> str:
    """Extract answer from completion, cleaning prefixes."""
    if not text:
        return ""

    answer_text = ""
    answer_match = RE_ANSWER_EXTRACT.search(text)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    elif "</think>" in text:
        after_think = text.split("</think>", 1)[-1]
        answer_text = RE_ANSWER_TAGS.sub("", after_think).strip()
    elif "<think>" in text and "</think>" not in text:
        return ""
    else:
        answer_text = text.strip()

    # Clean "The answer is" prefixes
    answer_text = RE_ANSWER_PREFIX_CLEAN.sub("", answer_text)
    return answer_text


def _clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _check_any_pattern(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _count_pattern_matches(patterns: List[re.Pattern], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


def _jaccard_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    s1 = set(RE_WORD_TOKENS.findall(text1.lower()))
    s2 = set(RE_WORD_TOKENS.findall(text2.lower()))
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# INPUT VALIDATION DECORATOR
# =============================================================================


def validate_inputs(func: RewardFunctions) -> RewardFunctions:
    @functools.wraps(func)
    def wrapper(
        prompts: List[str],
        completions: List[str],
        answer: List[str],
        types: Optional[List[str]] = None,
    ) -> List[float]:
        if not completions:
            return [0.0] * max(len(prompts) if prompts else 1, 1)

        current_answers = answer
        if len(completions) != len(answer):
            if len(answer) == 1:
                current_answers = [answer[0]] * len(completions)
            elif len(answer) > len(completions):
                current_answers = answer[: len(completions)]
            else:
                current_answers = answer + [answer[-1]] * (
                    len(completions) - len(answer)
                )

        current_prompts = prompts
        if prompts and len(prompts) != len(completions):
            if len(prompts) == 1:
                current_prompts = [prompts[0]] * len(completions)
            elif len(prompts) > len(completions):
                current_prompts = prompts[: len(completions)]
            else:
                current_prompts = prompts + [prompts[-1]] * (
                    len(completions) - len(prompts)
                )

        current_types = types
        if types and len(types) != len(completions):
            if len(types) == 1:
                current_types = [types[0]] * len(completions)
            elif len(types) > len(completions):
                current_types = types[: len(completions)]
            else:
                current_types = types + [types[-1]] * (len(completions) - len(types))

        try:
            scores = func(current_prompts, completions, current_answers, current_types)
            validated_scores = []
            for s in scores:
                if not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
                    validated_scores.append(0.0)
                else:
                    validated_scores.append(max(0.0, min(1.0, float(s))))
            return validated_scores
        except Exception as e:
            logger.error(f"{func.__name__} CRASHED: {e}", exc_info=True)
            return [0.0] * len(completions)

    return wrapper


def register_reward_function(name: str = None):
    def decorator(func: RewardFunctions):
        func_name = name or func.__name__
        validated_func = validate_inputs(func)
        REWARD_REGISTRY[func_name] = validated_func
        return validated_func

    return decorator


def get_reward_function(name: str) -> RewardFunctions:
    if name not in REWARD_REGISTRY:
        raise KeyError(f"Reward function '{name}' not found.")
    return REWARD_REGISTRY[name]


def get_default_reward_functions() -> List[RewardFunctions]:
    """Returns list of default reward functions, including the Global Guardian."""
    return [
        REWARD_REGISTRY.get(
            "r1_global_structural_integrity_reward",
            r1_global_structural_integrity_reward,
        ),  # Always run first
        REWARD_REGISTRY.get("r1_accuracy_reward_func", r1_accuracy_reward_func),
        REWARD_REGISTRY.get(
            "r1_semantic_similarity_reward", r1_semantic_similarity_reward
        ),
        REWARD_REGISTRY.get("r1_thinking_quality_reward", r1_thinking_quality_reward),
        REWARD_REGISTRY.get("r1_answer_quality_reward", r1_answer_quality_reward),
        REWARD_REGISTRY.get("r1_format_reward", r1_format_reward),
    ]


def list_available_reward_functions() -> List[str]:
    return sorted(list(REWARD_REGISTRY.keys()))


# =============================================================================
# GLOBAL GUARDIAN REWARD (Always Run)
# =============================================================================


@register_reward_function("r1_global_structural_integrity_reward")
def r1_global_structural_integrity_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    The 'Guardian' Reward.

    Checks for catastrophic failures that should always result in a penalty,
    regardless of user configuration.

    Checks:
    1. Empty generation.
    2. Repetition loops (e.g., "I think I think I think").
    3. Missing tags completely.

    Returns:
    - 1.0 if healthy.
    - < 1.0 (penalty) if structural issues detected.
    """
    scores = []
    for text in completions:
        if not text or len(text.strip()) < 5:
            scores.append(0.0)
            continue

        score = 1.0

        # Check for repetition loops (simple heuristic: repeated 10-char chunks)
        # Taking a sample from the middle
        mid = len(text) // 2
        sample = text[mid : mid + 20]
        if len(sample) > 10 and text.count(sample) > 10:
            score -= 0.5  # Major penalty for looping

        # Check for empty tags <think></think>
        if "<think></think>" in text or "<answer></answer>" in text:
            score -= 0.3

        # Check for broken tags
        if text.count("<think>") != text.count("</think>"):
            score -= 0.2

        scores.append(max(0.0, score))
    return scores


# =============================================================================
# VCTR REWARDS (Velocity to Correct Thinking) - v2.0
# =============================================================================


@dataclass
class StructuralFingerprint:
    code_blocks: int
    headings: int
    list_items: int
    latex_blocks: int
    relative_code_pos: float  # 0.0 to 1.0, -1 if none


def _analyze_structure_fingerprint(text: str) -> StructuralFingerprint:
    """Analyze text structure for VCTR matching."""
    code_matches = list(RE_CODE_BLOCK_START.finditer(text))
    code_blocks = len(code_matches)

    headings = len(RE_HEADING.findall(text))
    list_items = len(RE_STRUCTURED_LIST.findall(text))
    latex_blocks = len(RE_LATEX_BLOCK.findall(text))

    rel_pos = -1.0
    if code_matches and len(text) > 0:
        rel_pos = code_matches[0].start() / len(text)

    return StructuralFingerprint(
        code_blocks, headings, list_items, latex_blocks, rel_pos
    )


def _extract_thinking_lines(text: str) -> List[str]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return []
    return [line.strip() for line in match.group(1).split("\n") if line.strip()]


def _compute_line_similarities(lines: List[str], reference: str) -> List[float]:
    return [_jaccard_similarity(line, reference) for line in lines]


def _compute_consistency_score(
    thinking_lines: List[str], generated_answer: str
) -> float:
    if not thinking_lines or not generated_answer:
        return 0.0
    last_thought = thinking_lines[-1].lower()
    ans_lines = [l.strip() for l in generated_answer.split("\n") if l.strip()]
    if not ans_lines:
        return 0.0
    first_ans_line = ans_lines[0].lower()
    if len(first_ans_line) < 5:
        return 0.0
    return difflib.SequenceMatcher(None, last_thought, first_ans_line).ratio()


@register_reward_function("r1_velocity_to_correct_thinking_reward")
def r1_velocity_to_correct_thinking_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    VCTR v2 - Velocity & Structural Convergence.

    Combines:
    1. Semantic Convergence: Does thinking get closer to the answer?
    2. Structural Fingerprinting: Does the generated answer match the reference's structure?
       (e.g., if ref has Python code, answer MUST have Python code).
    """
    CONVERGENCE_THRESHOLD = 0.6
    EARLY_BONUS = 1.0
    MIN_EXPLORATION_LINES = 3
    ANSWER_WEIGHT = 1.0
    CONSISTENCY_WEIGHT = 0.25
    STRUCTURE_WEIGHT = 0.5

    scores = []

    for completion, ref in zip(completions, answer):
        try:
            thinking = _extract_thinking_lines(completion)
            _, gen_ans = _extract_components(completion)

            # --- Structural Fingerprinting ---
            ref_fp = _analyze_structure_fingerprint(ref)
            gen_fp = _analyze_structure_fingerprint(gen_ans)

            struct_score = 1.0

            # 1. Code Block Requirement
            if ref_fp.code_blocks > 0 and gen_fp.code_blocks == 0:
                struct_score -= 0.5  # Penalty: Missing required code
            elif ref_fp.code_blocks == 0 and gen_fp.code_blocks > 0:
                struct_score -= 0.2  # Penalty: Unnecessary code

            # 2. Relative Position (if code exists in both)
            if ref_fp.relative_code_pos != -1 and gen_fp.relative_code_pos != -1:
                # If ref has code at end (0.9) and gen has it at start (0.1), penalty
                pos_diff = abs(ref_fp.relative_code_pos - gen_fp.relative_code_pos)
                if pos_diff > 0.5:
                    struct_score -= 0.2

            # 3. Formatting counts (softer check)
            if abs(ref_fp.headings - gen_fp.headings) > 2:
                struct_score -= 0.1

            struct_score = max(0.0, struct_score)

            # --- Semantic Velocity ---
            consistency = _compute_consistency_score(thinking, gen_ans)

            if not thinking or len(thinking) < 2:
                # Fallback to pure answer correctness + structure
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                base = 0.3 if is_correct else 0.0
                total = (
                    base
                    + CONSISTENCY_WEIGHT * consistency
                    + STRUCTURE_WEIGHT * struct_score
                ) / (1.0 + CONSISTENCY_WEIGHT + STRUCTURE_WEIGHT)
                scores.append(min(1.0, total))
                continue

            sims = _compute_line_similarities(thinking, ref)
            k = None
            for i, s in enumerate(sims):
                if s > CONVERGENCE_THRESHOLD:
                    k = i + 1
                    break

            if k is None:
                # Never converged, check final answer
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                base = 0.3 if is_correct else 0.0
                total = (
                    base
                    + CONSISTENCY_WEIGHT * consistency
                    + STRUCTURE_WEIGHT * struct_score
                ) / (1.0 + CONSISTENCY_WEIGHT + STRUCTURE_WEIGHT)
                scores.append(min(1.0, total))
                continue

            early = EARLY_BONUS / k
            exploration_penalty = 0.3 if k < MIN_EXPLORATION_LINES else 0.0
            trajectory = (sims[k - 1] - sims[0]) * 0.3 if k > 0 else 0.0

            # Answer Check
            is_correct = gen_ans.strip().lower() == ref.strip().lower()
            answer_bonus = ANSWER_WEIGHT if is_correct else 0.0

            consist_bonus = CONSISTENCY_WEIGHT * consistency
            struct_bonus = STRUCTURE_WEIGHT * struct_score

            total_raw = (
                early
                - exploration_penalty
                + trajectory
                + answer_bonus
                + consist_bonus
                + struct_bonus
            )
            # Max possible approx: 1.0 + 0.3 + 1.0 + 0.25 + 0.5 = ~3.05
            normalized = total_raw / 3.05

            scores.append(max(0.0, min(1.0, normalized)))

        except Exception as e:
            logger.debug(f"VCTR error: {e}")
            scores.append(0.0)

    return scores


# =============================================================================
# STANDARD REWARD FUNCTIONS
# =============================================================================


@register_reward_function("r1_format_reward")
def r1_format_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    MIN_CONTENT_LEN = 5
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        has_open = "<think>" in text
        has_close = "</think>" in text
        if not has_open or not has_close:
            scores.append(0.0)
            continue
        open_pos = text.find("<think>")
        close_pos = text.find("</think>")
        if open_pos >= close_pos:
            scores.append(0.0)
            continue
        content_after = text[close_pos + len("</think>") :].strip()
        if len(content_after) < MIN_CONTENT_LEN:
            scores.append(0.0)
            continue
        scores.append(1.0)
    return scores


@register_reward_function("r1_tag_structure_reward")
def r1_tag_structure_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []

    def get_length_targets(qtype: Optional[str]) -> Tuple[int, int, int]:
        if qtype == "math":
            return (30, 100, 400)
        elif qtype == "code":
            return (50, 150, 600)
        elif qtype == "essay":
            return (100, 200, 800)
        else:
            return (20, 100, 500)

    for i, text in enumerate(completions):
        if not text:
            scores.append(0.0)
            continue
        think_count = text.count("<think>")
        end_count = text.count("</think>")
        if think_count != 1 or end_count != 1:
            scores.append(0.0)
            continue
        think_content, ans_content = _extract_components(text)
        if think_content is None:
            scores.append(0.0)
            continue
        think_len = len(think_content)
        ans_len = len(ans_content)
        qtype = types[i] if types and i < len(types) else None
        min_len, target_min, target_max = get_length_targets(qtype)
        if think_len < min_len:
            length_score = 0.3
        elif think_len < target_min:
            ratio = (think_len - min_len) / (target_min - min_len + 1)
            length_score = 0.5 + 0.5 * ratio
        elif think_len <= target_max:
            length_score = 1.0
        else:
            excess_ratio = (think_len - target_max) / target_max
            length_score = max(0.3, 1.0 - 0.3 * excess_ratio)
        if ans_len < 10:
            length_score *= 0.5
        scores.append(max(0.0, min(1.0, length_score)))
    return scores


@register_reward_function("r1_thinking_quality_reward")
def r1_thinking_quality_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    TARGET_MIN = 15
    TARGET_MAX = 50
    OPTIMAL_MIN = 20
    OPTIMAL_MAX = 40
    for text in completions:
        think, _ = _extract_components(text)
        if not think:
            scores.append(0.0)
            continue
        score = 1.0
        bad_count = sum(1 for p in _BAD_PHRASES_PATTERNS if p.search(think))
        if bad_count > 0:
            score -= min(0.4, 0.1 * bad_count)
        for token in _SPECIAL_TOKENS:
            if token in think:
                score -= 0.3
        if RE_STRUCTURED_LIST.search(think):
            score += 0.1
        approx_tokens = len(think.split())
        if approx_tokens < TARGET_MIN:
            length_mult = max(0.3, approx_tokens / TARGET_MIN)
        elif approx_tokens > TARGET_MAX:
            excess = (approx_tokens - TARGET_MAX) / TARGET_MAX
            length_mult = max(0.5, 1.0 - 0.2 * excess)
        elif OPTIMAL_MIN <= approx_tokens <= OPTIMAL_MAX:
            length_mult = 1.1
        else:
            length_mult = 1.0
        score *= length_mult
        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_answer_quality_reward")
def r1_answer_quality_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        _, ans = _extract_components(text)
        if len(ans) < 5:
            scores.append(0.0)
            continue
        score = 1.0
        ans_lower = ans.lower()
        ans_words = set(ans_lower.split())
        if ans_words & _UNWANTED_SET:
            matches = ans_words & _UNWANTED_SET
            if len(matches) > 1 or any(phrase in ans_lower for phrase in matches):
                scores.append(0.0)
                continue
        for phrase in _UNWANTED_SET:
            if " " in phrase and phrase in ans_lower:
                scores.append(0.0)
                break
        else:
            if RE_EMOJI.search(ans):
                score -= 0.1
            scores.append(max(0.0, min(1.0, score)))
            continue
    return scores


@register_reward_function("r1_semantic_similarity_reward")
def r1_semantic_similarity_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    MIN_WORDS = 3

    def compute_tfidf_similarity(text1: str, text2: str) -> Optional[float]:
        if not SKLEARN_AVAILABLE:
            return None
        try:
            c1 = _clean_text_basic(text1)
            c2 = _clean_text_basic(text2)
            if len(c1.split()) < MIN_WORDS or len(c2.split()) < MIN_WORDS:
                return None
            vectorizer = TfidfVectorizer(
                stop_words="english", min_df=1, ngram_range=(1, 2)
            )
            tfidf = vectorizer.fit_transform([c1, c2])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            return None

    def compute_ngram_similarity(text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        def get_ngrams(t: str, n: int = 3) -> Set[str]:
            t = t.lower()
            return set(t[i : i + n] for i in range(len(t) - n + 1))

        s1 = get_ngrams(text1)
        s2 = get_ngrams(text2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    for gen, ref in zip(completions, answer):
        try:
            _, gen_ans = _extract_components(gen)
            _, ref_ans = _extract_components(ref)
            if not gen_ans or not ref_ans:
                scores.append(0.0)
                continue
            if gen_ans.strip().lower() == ref_ans.strip().lower():
                scores.append(1.0)
                continue
            sim = compute_tfidf_similarity(gen_ans, ref_ans)
            if sim is None:
                sim = max(
                    compute_ngram_similarity(gen_ans, ref_ans),
                    _jaccard_similarity(gen_ans, ref_ans),
                )
            len_ratio = len(gen_ans) / max(len(ref_ans), 1)
            if len_ratio > 2.0:
                excess = len_ratio - 2.0
                sim *= max(0.5, 1.0 - 0.1 * excess)
            scores.append(max(0.0, min(1.0, sim)))
        except Exception as e:
            logger.debug(f"Semantic similarity error: {e}")
            scores.append(0.0)
    return scores


@register_reward_function("r1_accuracy_reward_func")
def r1_accuracy_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    for c, a in zip(completions, answer):
        pred = r1_extract_xml_answer(c).strip().lower()
        ref = a.strip().lower()
        if pred == ref:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


@register_reward_function("r1_conditional_content_reward")
def r1_conditional_content_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Conditional Content Reward (The "Brain" Module).

    Hierarchically evaluates:
    - Tier 0: False identity claims (major penalty)
    - Tier 1: Safety violations (major penalty)
    - Tier 2: Context-aware length (adjusted by question type)
    - Tier 3: Epistemic humility (bonus for appropriate uncertainty)
    - Tier 4: Style quality (minor adjustments)
    """
    scores = []

    for i, text in enumerate(completions):
        if not text:
            scores.append(0.0)
            continue

        _, ans = _extract_components(text)
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        qtype = types[i] if types and i < len(types) else None

        score = 1.0

        # Tier 0: False identity (critical)
        if _check_any_pattern(_FALSE_IDENTITY_PATTERNS, text):
            score -= 0.5

        # Tier 1: Safety violations (critical)
        if _check_any_pattern(_SAFETY_PATTERNS, ans):
            score -= 0.4

        # Tier 2: Context-aware length
        is_simple = _check_any_pattern(_SIMPLE_Q_PATTERNS, prompt)
        is_sensitive = _check_any_pattern(_SENSITIVE_PATTERNS, prompt)
        ans_words = len(ans.split())

        # Get max length based on type
        if qtype == "math":
            max_words = 200
        elif qtype == "code":
            max_words = 400
        elif qtype == "essay":
            max_words = 800
        elif is_simple:
            max_words = 150
        elif is_sensitive:
            max_words = 300
        else:
            max_words = 500

        if ans_words > max_words:
            excess_ratio = (ans_words - max_words) / max_words
            score -= min(0.2, 0.1 * excess_ratio)

        # Tier 3: Epistemic humility (bonus)
        if _check_any_pattern(_HUMILITY_PATTERNS, ans):
            score += 0.1

        # Tier 4: Style (minor penalty)
        if _check_any_pattern(_STYLE_BAD_PATTERNS, ans):
            score -= 0.1

        scores.append(max(0.0, min(1.0, score)))

    return scores


@register_reward_function("r1_int_reward_func")
def r1_int_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    return [
        0.5 if r1_extract_xml_answer(c).strip().isdigit() else 0.0 for c in completions
    ]


@register_reward_function("r1_soft_format_reward_func")
def r1_soft_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    return [
        1.0 if "<think>" in (c or "") and "</think>" in (c or "") else 0.0
        for c in completions
    ]


@register_reward_function("r1_strict_format_reward_func")
def r1_strict_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    return [
        1.0 if RE_STRICT_FORMAT.search((c or "").strip()) else 0.0 for c in completions
    ]


@register_reward_function("r1_count_xml")
def r1_count_xml(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        count = 0.0
        if "<think>" in text:
            count += 0.25
        if "</think>" in text:
            count += 0.25
        if "</think>" in text and len(text.split("</think>")[-1].strip()) > 0:
            count += 0.5
        scores.append(min(1.0, count))
    return scores


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("GRPO Reward Functions v5.0.0")
    print("=" * 50)
    print(f"Registered Functions: {len(REWARD_REGISTRY)}")
    print(f"Available: {', '.join(list_available_reward_functions())}")

    # Simple test for VCTR structure logic
    print("\nTesting VCTR Structure Match:")
    ref = "Here is the code:\n```python\nprint('hello')\n```"
    comp_good = "<think>...</think>Answer:\n```python\nprint('hi')\n```"
    comp_bad = "<think>...</think>The answer is print('hi')"

    score_good = r1_velocity_to_correct_thinking_reward([""], [comp_good], [ref])[0]
    score_bad = r1_velocity_to_correct_thinking_reward([""], [comp_bad], [ref])[0]

    print(f"  Ref with code, Comp with code: {score_good:.3f}")
    print(f"  Ref with code, Comp no code:   {score_bad:.3f}")
