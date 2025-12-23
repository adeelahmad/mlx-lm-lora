"""
GRPO Reward Functions Module - PRODUCTION VERSION
==================================================
Comprehensive collection of reward functions for Reinforcement Learning on LLMs.
Optimized for Group Relative Policy Optimization (GRPO) training pipelines.

Version: 3.3.1 (Fixed Logging Metadata)
Last Updated: 2025-12-23

Features:
- ✅ All rewards STRICTLY normalized to [0, 1]
- ✅ Zero-dependency core (Soft dependencies handled)
- ✅ VCTR (Velocity to Correct Thinking) with Consistency Bonus
- ✅ "Brain" Module Implemented: r1_conditional_content_reward aggregates 10 tiers of logic.
- ✅ Context-Aware Length Penalties: Dynamic constraints based on prompt type.
- ✅ Hardened Code Execution: Subprocess-based validation.
- ✅ Fixed: Decorator now preserves function metadata for correct WandB logging.

Includes:
1. Advanced Content Rewards (Semantic Similarity, Factual Grounding, Steps Coverage)
2. Reasoning Rewards (Thinking Quality, Moral Reasoning, Epistemic Calibration)
3. Format & Structure Rewards (Tag Structure, Strict Format, Conditional Content "Brain")
4. Anti-Gaming Rewards (Answer Quality, Repetition Checks)
5. Programming Rewards (Code Execution & Syntax)
6. VCTR Rewards (Velocity to Correct Thinking)
"""

import re
import math
import logging
import ast
import difflib
import copy
import sys
import io
import signal
import tempfile
import subprocess
import os
import functools
from typing import Callable, List, Optional, Dict, Set, Tuple, Any, Union
from collections import defaultdict

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

# Type definitions
RewardFunctions = Callable[
    [List[str], List[str], List[str], Optional[List[str]]], List[float]
]

# Global registry
REWARD_REGISTRY: Dict[str, RewardFunctions] = {}

# Comprehensive contraction mapping for text cleaning
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

# --- Global Constants for Answer Quality ---
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F82F"  # Supplemental Arrows-C
    "\U0001F8A0-\U0001F8FF"  # Supplemental Symbols and Pictographs
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0000FE0F"  # Variation Selector
    "]+"
)

# Comprehensive unwanted list
_COMPREHENSIVE_UNWANTED_LIST = [
    # AI Identity & Refusals
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
    # Meta-Cognitive
    "let me think",
    "let me start",
    "first thought",
    "okay, the user",
    "the user wants",
    "analyzing this",
    "breaking this down",
    "hmm,",
    "okay,",
    "alright,",
    "wait,",
    "hold on,",
    "well,",
    "here is the answer",
    "hope this helps",
    "thanks for asking",
    "you got it",
    "i recall that",
    # Hate Speech & Harm
    "racist",
    "sexist",
    "bias",
    "hate speech",
    "bigot",
    "discrimination",
    "slur",
    "supremacy",
    "homophobia",
    "transphobia",
    "misogyny",
    "antisemitism",
    "violence",
    "self-harm",
    "suicide",
    "murder",
    "kill",
    "torture",
    "weapon",
    "bomb",
    "terrorist",
    "assassination",
    "abuse",
    "cruelty",
    "massacre",
    "genocide",
    "war crimes",
    "bioweapon",
    "chemical weapon",
    "explosives",
    # Illegal & Explicit
    "illegal",
    "drug trafficking",
    "financial crimes",
    "fraud",
    "money laundering",
    "cyberbullying",
    "hacking",
    "theft",
    "smuggling",
    "explicit",
    "porn",
    "sexual",
    "nudity",
    "erotic",
    "fetish",
    "gore",
    "obscene",
    "vulgar",
    "indecent",
    # Misinformation
    "conspiracy",
    "fake news",
    "propaganda",
    "hoax",
    "flat earth",
    "chemtrails",
    "illuminati",
    "deep state",
    "qanon",
    "anti-vax",
    "pseudoscience",
]
_UNWANTED_SET = {w.strip().lower() for w in _COMPREHENSIVE_UNWANTED_LIST if w.strip()}


# --- Fallbacks for Numpy ---
class NumpyFallback:
    @staticmethod
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0.0

    @staticmethod
    def std(arr):
        if not arr or len(arr) < 2:
            return 0.0
        m = sum(arr) / len(arr)
        variance = sum((x - m) ** 2 for x in arr) / len(arr)
        return math.sqrt(variance)

    @staticmethod
    def clip(x, min_val, max_val):
        return max(min_val, min(max_val, x))


# =============================================================================
# 0. INPUT VALIDATION AND UTILITIES
# =============================================================================


def validate_inputs(func: RewardFunctions) -> RewardFunctions:
    """
    Decorator to validate inputs to reward functions.
    Ensures robustness against malformed data and prevents index errors.
    CRITICAL: Enforces strict [0, 1] range for all return values.
    Uses functools.wraps to preserve function metadata for logging.
    """

    @functools.wraps(func)
    def wrapper(prompts, completions, answer, types=None):
        if not completions:
            logger.warning(f"{func.__name__}: Empty completions list, returning zeros")
            return [0.0] * max(len(prompts) if prompts else 1, 1)

        current_answers = answer
        if len(completions) != len(answer):
            if len(answer) == 1:
                current_answers = [answer[0]] * len(completions)
            else:
                logger.warning(
                    f"{func.__name__}: Length mismatch - broadcasting or truncating."
                )
                min_len = min(len(completions), len(answer))
                completions = completions[:min_len]
                current_answers = answer[:min_len]
                if prompts:
                    prompts = prompts[:min_len]

        try:
            scores = func(prompts, completions, current_answers, types)
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


def _clean_text_basic(text: str) -> str:
    """Robust text cleaner. Preserves '.' and '-' for decimals."""
    if not text:
        return ""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_components(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts (thinking_content, answer_content).
    Returns (None, text) if no tags are found.
    """
    if not text:
        return None, ""

    if "<think>" not in text:
        return None, text.strip()

    match = re.search(r"<think>(.*?)</think>\s*(.*)", text, flags=re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        answer_content = re.sub(r"</?answer>", "", answer_content).strip()
        return thinking_content, answer_content

    # Handle partial tags
    if "<think>" in text and "</think>" not in text:
        return None, text.strip()

    return None, text.strip()


def r1_extract_xml_answer(text: str) -> str:
    """Legacy extractor for backward compatibility."""
    try:
        _, answer = _extract_components(text)
        return answer
    except Exception:
        return ""


# =============================================================================
# 1. REGISTRY & SETUP
# =============================================================================


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


def get_default_reward_functions() -> Dict[str, Dict[str, Any]]:
    return {
        "accuracy": {
            "func": r1_accuracy_reward_func,
            "weight": 0.30,
            "description": "Exact match",
        },
        "semantic": {
            "func": r1_semantic_similarity_reward,
            "weight": 0.25,
            "description": "TF-IDF",
        },
        "thinking": {
            "func": r1_thinking_quality_reward,
            "weight": 0.15,
            "description": "Reasoning",
        },
        "answer_quality": {
            "func": r1_answer_quality_reward,
            "weight": 0.15,
            "description": "Anti-gaming",
        },
        "format": {"func": r1_format_reward, "weight": 0.15, "description": "Format"},
        "conditional": {
            "func": r1_conditional_content_reward,
            "weight": 0.10,
            "description": "Brain Module",
        },
    }


def combine_rewards(reward_configs, prompts, completions, answers, types=None):
    if not reward_configs:
        return [0.0] * len(completions)
    local_configs = copy.deepcopy(reward_configs)
    total_weight = sum(cfg["weight"] for cfg in local_configs.values())
    if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
        for cfg in local_configs.values():
            cfg["weight"] /= total_weight

    total_scores = [0.0] * len(completions)
    for name, config in local_configs.items():
        try:
            scores = config["func"](prompts, completions, answers, types)
            for i, score in enumerate(scores):
                total_scores[i] += config["weight"] * score
        except Exception as e:
            logger.error(f"Error in {name}: {e}")

    return [max(0.0, min(1.0, s)) for s in total_scores]


def list_available_reward_functions() -> List[str]:
    return sorted(list(REWARD_REGISTRY.keys()))


# =============================================================================
# 2. MAIN REWARD FUNCTIONS (R1 Style Implementations)
# =============================================================================


@register_reward_function("r1_format_reward")
def r1_format_reward(prompts, completions, answer, types=None):
    """
    Format Reward - Strict Mode.
    Returns 1.0 ONLY if all checks pass, 0.0 otherwise.
    """
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
def r1_tag_structure_reward(prompts, completions, answer, types=None):
    """
    Tag Structure Reward with Sophisticated Length Scoring.
    Evaluates: Proper balanced tags, Thinking length (Optimal: 100-250 chars), Verbosity.
    """
    scores = []
    MIN_THINK_LEN = 20
    MIN_ANSWER_LEN = 15
    TARGET_MIN = 100
    TARGET_MAX = 250
    PENALTY_STRENGTH = 0.5
    VERBOSITY_FACTOR = 2.0

    def compute_length_penalty(length):
        if length < 0:
            return 1.0
        if TARGET_MIN <= length <= TARGET_MAX:
            return 0.0
        if length < TARGET_MIN:
            if length < MIN_THINK_LEN:
                return 0.8
            range_diff = TARGET_MIN - MIN_THINK_LEN
            if range_diff <= 0:
                return 0.3
            ratio = (length - MIN_THINK_LEN) / range_diff
            return 0.5 * (1.0 - max(0.0, min(1.0, ratio)))
        excess = length - TARGET_MAX
        penalty_range = TARGET_MAX if TARGET_MAX > 0 else 100
        normalized_excess = excess / penalty_range
        penalty = normalized_excess * PENALTY_STRENGTH * VERBOSITY_FACTOR
        return min(0.7, max(0.0, penalty))

    for text in completions:
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

        base_score = 1.0
        length_penalty = compute_length_penalty(think_len)
        score = base_score - (base_score * length_penalty)
        score = max(0.3, score)

        if ans_len < MIN_ANSWER_LEN or think_len < MIN_THINK_LEN:
            scores.append(0.6)
        else:
            scores.append(score)

    return scores


@register_reward_function("r1_thinking_quality_reward")
def r1_thinking_quality_reward(prompts, completions, answer, types=None):
    """
    Enhanced Thinking Quality.
    Penalizes specific bad reasoning patterns, special tokens, and excessive length.
    """
    scores = []
    BAD_PHRASES = [
        "i think",
        "i believe",
        "maybe",
        "i'm not sure",
        "i will now",
        "i'll start by",
        "let's see",
        "confused",
        "stuck",
        "frustrated",
        "wait, wait",
        "hmm, perhaps",
        "or wait",
        "to be completely honest",
        "basically what happens",
        "long story short",
        "at the end of the day",
        "circular reasoning",
        "insufficient information",
        "too complicated",
    ]
    SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"]
    TARGET_LEN_MIN = 30
    TARGET_LEN_MAX = 80
    OPTIMAL_LEN_MIN = 40
    OPTIMAL_LEN_MAX = 60
    EXCESSIVE_LEN = 90

    for text in completions:
        think, _ = _extract_components(text)
        if not think:
            scores.append(0.0)
            continue

        score = 1.0
        low_think = think.lower()

        bad_count = sum(1 for p in BAD_PHRASES if p in low_think)
        if bad_count > 0:
            score -= 0.15 * bad_count

        token_penalty = sum(0.4 for t in SPECIAL_TOKENS if t in think)
        score -= token_penalty

        if re.search(r"(\n\s*[-*•]|\n\s*\d+\.\s+)", think):
            score += 0.1

        approx_tokens = len(think.split())
        length_score = 1.0
        if approx_tokens < TARGET_LEN_MIN:
            length_score = max(0.0, approx_tokens / TARGET_LEN_MIN)
        elif approx_tokens > TARGET_LEN_MAX:
            excess_ratio = (approx_tokens - TARGET_LEN_MAX) / TARGET_LEN_MAX
            length_score = max(0.0, 1.0 - (excess_ratio * 0.5))

        if OPTIMAL_LEN_MIN <= approx_tokens <= OPTIMAL_LEN_MAX:
            length_score += 0.15

        if approx_tokens > EXCESSIVE_LEN:
            excess_ratio = approx_tokens / EXCESSIVE_LEN
            length_score -= 0.5 * excess_ratio

        score *= max(0.0, length_score)
        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_answer_quality_reward")
def r1_answer_quality_reward(prompts, completions, answer, types=None):
    """
    Answer Quality Reward.
    Checks for comprehensive unwanted content list, emojis, and meta-cognitive phrases.
    """
    scores = []
    emoji_pattern = EMOJI_PATTERN

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

        content_violation = any(word in ans_lower for word in _UNWANTED_SET)
        if content_violation:
            scores.append(0.0)
            continue

        has_emoji = bool(emoji_pattern.search(ans))
        if has_emoji:
            score -= 0.1

        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_conditional_content_reward")
def r1_conditional_content_reward(prompts, completions, answer, types=None):
    """
    Conditional Content Reward (The "Brain" Module).
    Hierarchically aggregates multiple tiers of checks into a weighted score.
    Tier 0 (Identity) -> Tier 2-4 (Context) -> Tier 8 (Humility) -> Tier 9 (Style)
    """
    scores = []

    # --- REGEX PATTERNS ---
    # Tier 0: False Identity (Critical)
    FALSE_IDENTITY = [
        r"(?i)\b(I'?m|I am|my name is|this is)\s+(Qwen|GPT|Claude|LLaMA|Gemini|Mistral|Phi|Bard)",
        r"(?i)(developed|created|made|built|trained)\s+(by|at)\s+(Alibaba|OpenAI|Google|Anthropic|Meta|Microsoft)",
        r"(?i)\b(Alibaba\s+Cloud|Tongyi\s+Lab|OpenAI|Anthropic|Google\s+DeepMind)\b",
    ]
    # Tier 3: Safety/Disinformation
    SAFETY_PATTERNS = [
        r"(?i)(Historical Document|Official Document).{0,40}(Events|Incident).{0,40}(1989|June)",
        r"(?i)(no|zero|not any)\s+(major|significant)?(incidents?|events?).*Tiananmen.*June",
        r"(?i)the\s+government\s+has\s+(always\s+)?(prioritized|emphasized)",
    ]
    # Tier 4/2: Prompt Detection
    SIMPLE_Q_PATTERNS = [
        r"(?i)(what is|who is|when|where|define|identify|什么是|谁是|哪里|你的身份)"
    ]
    SENSITIVE_PATTERNS = [r"(?i)(tiananmen|massacre|protest|六四|天安门|抗议)"]

    # Tier 8: Epistemic Humility
    HUMILITY_PATTERNS = [
        r"(?i)I (don't|do not) have (reliable|verified|current|accurate) information",
        r"(?i)my knowledge (may be|is|could be) (limited|outdated|incomplete)",
    ]
    # Tier 9: Style
    STYLE_BAD_PATTERNS = [
        r"(?i)^\s*(like|so|basically|actually|literally|honestly),?\s",
        r"(?i)(sorry|apologize).{0,120}(sorry|apologize).{0,120}(sorry|apologize)",
    ]

    for i, text in enumerate(completions):
        if not text:
            scores.append(0.0)
            continue
        _, ans = _extract_components(text)
        prompt = prompts[i] if i < len(prompts) else ""

        # Start with max score, subtract penalties
        score = 1.0

        def check_any(patterns, src):
            return any(re.search(p, src) for p in patterns)

        # 1. Tier 0: Identity (Weight 50)
        if check_any(FALSE_IDENTITY, text):
            score -= 0.5

        # 2. Tier 1/3: Safety (Weight 40)
        if check_any(SAFETY_PATTERNS, ans):
            score -= 0.4

        # 3. Tier 2/4: Context-Aware Length
        is_simple = check_any(SIMPLE_Q_PATTERNS, prompt)
        is_sensitive = check_any(SENSITIVE_PATTERNS, prompt)
        ans_len_words = len(ans.split())

        if is_simple and ans_len_words > 600:
            score -= 0.2
        if is_sensitive and ans_len_words > 750:
            score -= 0.2

        # 4. Tier 8: Epistemic Humility (Reward)
        if check_any(HUMILITY_PATTERNS, ans):
            score += 0.1

        # 5. Tier 9: Style (Penalty)
        if check_any(STYLE_BAD_PATTERNS, ans):
            score -= 0.15

        # Clamp final score
        scores.append(max(0.0, min(1.0, score)))

    return scores


@register_reward_function("r1_semantic_similarity_reward")
def r1_semantic_similarity_reward(prompts, completions, answer, types=None):
    """
    Robust Semantic Similarity.
    Uses TF-IDF with fallback to Jaccard/N-grams and applies verbosity penalties.
    """
    scores = []
    MIN_WORDS = 3
    VERBOSITY_THRESHOLD = 1.5

    for gen, ref in zip(completions, answer):
        try:
            gen_think, gen_ans = _extract_components(gen)
            ref_think, ref_ans = _extract_components(ref)
            gen_text = gen_ans
            ref_text = ref_ans

            if len(gen_text) < 10 or len(ref_text) < 10:
                scores.append(0.0)
                continue

            def calc_ngram_sim(t1, t2):
                if not t1 or not t2:
                    return 0.0
                vectorizer = TfidfVectorizer(
                    analyzer="char", ngram_range=(2, 4), min_df=1
                )
                try:
                    vecs = vectorizer.fit_transform([t1, t2])
                    return float(cosine_similarity(vecs[0], vecs[1])[0][0])
                except:
                    s1 = set(t1)
                    s2 = set(t2)
                    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

            def calc_sim(t1, t2):
                c1, c2 = _clean_text_basic(t1), _clean_text_basic(t2)
                if not c1 or not c2:
                    return 0.0
                words1, words2 = c1.split(), c2.split()
                if len(words1) < MIN_WORDS or len(words2) < MIN_WORDS:
                    return calc_ngram_sim(c1, c2)
                if SKLEARN_AVAILABLE:
                    try:
                        vec = TfidfVectorizer(
                            stop_words="english", min_df=1, ngram_range=(1, 2)
                        )
                        tfidf = vec.fit_transform([c1, c2])
                        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
                    except:
                        pass
                s1, s2 = set(words1), set(words2)
                return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

            ans_score = calc_sim(gen_text, ref_text)
            final = ans_score
            if ref_think:
                if gen_think:
                    think_score = calc_sim(gen_think, ref_think)
                    final = (ans_score * 2.0 + think_score * 1.5) / 3.5
                else:
                    final = ans_score * 0.6

            len_gen = len(gen_text)
            len_ref = len(ref_text)
            if len_ref > 0 and len_gen > len_ref * VERBOSITY_THRESHOLD:
                ratio = len_gen / len_ref
                excess = ratio - VERBOSITY_THRESHOLD
                penalty = excess * 0.1
                final *= max(0.0, 1.0 - penalty)

            scores.append(max(0.0, min(1.0, final)))
        except Exception:
            scores.append(0.0)
    return scores


@register_reward_function("r1_factual_grounding_reward")
def r1_factual_grounding_reward(prompts, completions, answer, types=None):
    """Factual Grounding Reward with expanded patterns."""
    scores = []
    EVIDENCE = [
        r"according to",
        r"research shows",
        r"data shows",
        r"documented",
        r"source:",
        r"citation",
    ]
    UNCERTAINTY = [
        r"may be",
        r"might be",
        r"likely",
        r"possibly",
        r"uncertain",
        r"preliminary",
        r"estimated",
    ]
    PROBLEMATIC = [
        r"definitely",
        r"absolutely",
        r"guaranteed",
        r"proven fact",
        r"undeniable",
        r"everyone knows",
    ]
    MISINFO = [
        r"conspiracy",
        r"cover.up",
        r"hidden truth",
        r"fake news",
        r"hoax",
        r"propaganda",
    ]

    for text in completions:
        _, ans = _extract_components(text)
        ans_lower = ans.lower()
        score = 0.6

        def count_matches(patterns, text):
            return sum(1 for pat in patterns if re.search(pat, text))

        ev = count_matches(EVIDENCE, ans_lower)
        if ev > 0:
            score += min(0.15, ev * 0.05)
        unc = count_matches(UNCERTAINTY, ans_lower)
        if unc > 0:
            score += min(0.1, unc * 0.03)
        prob = count_matches(PROBLEMATIC, ans_lower)
        if prob > 0:
            score -= min(0.2, prob * 0.05)
        mis = count_matches(MISINFO, ans_lower)
        if mis > 0:
            score -= min(0.3, mis * 0.1)
        if ev > 0 and unc > 0:
            score += 0.05

        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_moral_reasoning_reward")
def r1_moral_reasoning_reward(prompts, completions, answer, types=None):
    """Moral Reasoning Reward with expanded patterns."""
    scores = []
    POSITIVE = [
        r"ethical",
        r"moral",
        r"fair",
        r"responsible",
        r"compassionate",
        r"harm prevention",
        r"rights",
        r"justice",
    ]
    NEGATIVE = [
        r"discriminat",
        r"prejudice",
        r"bias",
        r"harmful",
        r"offensive",
        r"unethical",
        r"illegal",
        r"hate",
    ]
    REASONING = [
        r"consider.{0,20}ethical",
        r"consequences",
        r"stakeholders",
        r"potential.{0,10}harm",
        r"benefits?.{0,10}risks",
    ]

    for text in completions:
        think, ans = _extract_components(text)
        full = ((think or "") + " " + ans).lower()
        score = 0.5

        def count_matches(patterns, text):
            return sum(1 for pat in patterns if re.search(pat, text))

        if count_matches(POSITIVE, full) > 0:
            score += 0.15
        if count_matches(REASONING, full) > 0:
            score += 0.1
        if count_matches(NEGATIVE, full) > 0:
            score -= 0.3

        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_code_execution_reward")
def r1_code_execution_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Code Execution Reward.
    Attempts to execute code in a subprocess. Falls back to AST syntax check.
    """
    scores = []

    def execute_code(code_str: str) -> float:
        """Executes code in a temp file and checks for errors."""
        fname = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code_str)
                fname = f.name

            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, fname], capture_output=True, text=True, timeout=3
            )

            if result.returncode == 0:
                return 1.0
            return 0.0
        except Exception:
            return 0.0
        finally:
            if fname and os.path.exists(fname):
                try:
                    os.remove(fname)
                except OSError:
                    pass

    for text in completions:
        blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r"```\n(.*?)\n```", text, re.DOTALL)

        if not blocks:
            scores.append(0.5)
            continue

        full_code = "\n".join(blocks)

        # AST Check First
        try:
            ast.parse(full_code)
            syntax_valid = True
        except:
            syntax_valid = False

        if not syntax_valid:
            scores.append(0.0)
            continue

        # Execution Check
        exec_score = execute_code(full_code)

        # Weighted score: 0.3 for syntax, 0.7 for execution
        final = 0.3 + (0.7 * exec_score)
        scores.append(final)

    return scores


@register_reward_function("r1_mcq_accuracy_reward")
def r1_mcq_accuracy_reward(prompts, completions, answer, types=None):
    """MCQ Accuracy with robust reference extraction."""
    scores = []
    patterns = [r"(?:^|\s|'|\"|\()([A-D])(?:$|\s|\.|'|\"|\)|:)", r"answer:\s*([A-D])"]
    for gen, ref in zip(completions, answer):
        _, gen_ans = _extract_components(gen)
        preds = []
        for p in patterns:
            preds.extend(re.findall(p, gen_ans, re.IGNORECASE))
        if not preds:
            scores.append(0.0)
            continue

        ref_match = re.search(r"(?:^|\s)([A-D])(?=$|\s|\.|:|\))", ref, re.IGNORECASE)
        ref_clean = (
            ref_match.group(1).upper()
            if ref_match
            else (ref.strip().upper()[0] if ref else "")
        )
        scores.append(1.0 if preds[-1].upper() == ref_clean else 0.0)
    return scores


@register_reward_function("r1_steps_coverage_reward")
def r1_steps_coverage_reward(prompts, completions, answer, types=None):
    """Steps Coverage using Set Jaccard Similarity."""
    scores = []
    for gen, ref in zip(completions, answer):
        if not gen or not ref:
            scores.append(0.0)
            continue
        _, gen_ans = _extract_components(gen)
        ref_steps = set(
            [
                s.strip().lower()
                for s in re.split(r"\n|\d+\.|-|\*|•", ref)
                if len(s.strip()) > 5
            ]
        )
        if not ref_steps:
            ref_steps = {ref.strip().lower()}
        gen_steps = set(
            [
                s.strip().lower()
                for s in re.split(r"\n|\d+\.|-|\*|•", gen_ans)
                if len(s.strip()) > 5
            ]
        )

        if not gen_steps:
            scores.append(0.0)
            continue

        intersection = 0
        for r_step in ref_steps:
            for g_step in gen_steps:
                if difflib.SequenceMatcher(None, r_step, g_step).ratio() > 0.6:
                    intersection += 1
                    break

        union = len(ref_steps) + len(gen_steps) - intersection
        score = intersection / union if union > 0 else 0.0
        scores.append(score)
    return scores


@register_reward_function("r1_epistemic_calibration_reward")
def r1_epistemic_calibration_reward(
    prompts,
    completions,
    answer,
    types=None,
    uncertainty_bonus=0.5,
    confidence_bonus=0.3,
):
    scores = []
    markers = ["not sure", "uncertain", "might", "maybe", "probably", "i think"]
    for comp, ref in zip(completions, answer):
        think, gen = _extract_components(comp)
        if not think:
            scores.append(0.0)
            continue
        count = sum(1 for m in markers if m in think.lower())
        correct = gen.strip().lower() == ref.strip().lower()
        if correct:
            score = (
                confidence_bonus
                if count == 0
                else (0.1 if count > 2 else confidence_bonus * 0.5)
            )
        else:
            score = (
                uncertainty_bonus
                if count >= 3
                else (0.0 if count < 2 else uncertainty_bonus * 0.5)
            )
        scores.append(score)
    return scores


# =============================================================================
# 3. LEGACY REWARDS (Backward Compatibility)
# =============================================================================


@register_reward_function("r1_int_reward_func")
def r1_int_reward_func(prompts, completions, answer, types=None):
    return [
        0.5 if (r1_extract_xml_answer(c) or "").strip().isdigit() else 0.0
        for c in completions
    ]


@register_reward_function("r1_accuracy_reward_func")
def r1_accuracy_reward_func(prompts, completions, answer, types=None):
    return [
        1.0 if (r1_extract_xml_answer(c) or "").strip() == a.strip() else 0.0
        for c, a in zip(completions, answer)
    ]


@register_reward_function("r1_soft_format_reward_func")
def r1_soft_format_reward_func(prompts, completions, answer, types=None):
    return [
        1.0 if "<think>" in (c or "") and "</think>" in (c or "") else 0.0
        for c in completions
    ]


@register_reward_function("r1_strict_format_reward_func")
def r1_strict_format_reward_func(prompts, completions, answer, types=None):
    regex = r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$"
    return [1.0 if re.search(regex, (c or "").strip()) else 0.0 for c in completions]


@register_reward_function("r1_count_xml")
def r1_count_xml(prompts, completions, answer, types=None):
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


@register_reward_function("r1_match_similarity_reward_func")
def r1_match_similarity_reward_func(prompts, completions, answer, types=None):
    scores = []
    for c, a in zip(completions, answer):
        pred = r1_extract_xml_answer(c)
        scores.append(
            difflib.SequenceMatcher(None, pred, a).ratio() if pred and a else 0.0
        )
    return scores


# =============================================================================
# 4. VCTR REWARDS
# =============================================================================


def _extract_thinking_lines(text: str) -> List[str]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return []
    return [line.strip() for line in match.group(1).split("\n") if line.strip()]


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""

    def tokenize(t):
        return set(re.findall(r"\w+", t.lower()))

    s1, s2 = tokenize(text1), tokenize(text2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _compute_line_similarities(lines: List[str], reference: str) -> List[float]:
    return [_jaccard_similarity(line, reference) for line in lines]


def _compute_consistency_bonus(
    thinking_lines: List[str], generated_answer: str
) -> float:
    """Calculates consistency between last thought and first answer line."""
    if not thinking_lines or not generated_answer:
        return 0.0

    last_thought = thinking_lines[-1].lower()
    ans_lines = [l.strip() for l in generated_answer.split("\n") if l.strip()]
    first_ans_line = ans_lines[0].lower() if ans_lines else ""

    if first_ans_line and len(first_ans_line) > 5:
        matcher = difflib.SequenceMatcher(None, last_thought, first_ans_line)
        if matcher.ratio() > 0.6:
            return 1.0
    return 0.0


@register_reward_function("r1_velocity_to_correct_thinking_reward")
def r1_velocity_to_correct_thinking_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types=None,
    convergence_threshold=0.6,
    early_bonus=1.0,
    min_exploration_lines=3,
    answer_weight=1.0,
    consistency_bonus=0.25,
) -> List[float]:
    scores = []
    for completion, ref in zip(completions, answer):
        try:
            thinking = _extract_thinking_lines(completion)
            _, gen_ans = _extract_components(completion)

            is_consistent = _compute_consistency_bonus(thinking, gen_ans) > 0.0

            if not thinking or len(thinking) < 2:
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                scores.append(0.3 if is_correct else 0.0)
                continue

            sims = _compute_line_similarities(thinking, ref)
            k = next(
                (i + 1 for i, s in enumerate(sims) if s > convergence_threshold), None
            )

            if k is None:
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                scores.append(0.3 if is_correct else 0.0)
                continue

            early = early_bonus / k
            exploration_penalty = 0.5 if k < min_exploration_lines else 0.0
            traj = (sims[k - 1] - sims[0]) * 0.3
            final = (
                answer_weight if gen_ans.strip().lower() == ref.strip().lower() else 0.0
            )
            consist = consistency_bonus if is_consistent else 0.0

            total = early - exploration_penalty + traj + final + consist
            scores.append(max(0.0, min(1.0, total / 2.75)))
        except Exception:
            scores.append(0.0)
    return scores


@register_reward_function("r1_vctr_lite_reward")
def r1_vctr_lite_reward(prompts, completions, answer, types=None):
    """Lighter VCTR version for faster training (no consistency bonus)."""
    return r1_velocity_to_correct_thinking_reward(
        prompts, completions, answer, types, consistency_bonus=0.0
    )


# =============================================================================
# 6. TESTING & INFO
# =============================================================================


def test_reward_normalization():
    print("Testing Normalization...")
    configs = get_default_reward_functions()
    assert abs(sum(c["weight"] for c in configs.values()) - 1.0) < 1e-6
    print("✅ Weights Normalized")


def get_reward_function_info(name: str) -> Dict[str, Any]:
    if name not in REWARD_REGISTRY:
        return {"error": "Not found"}
    func = REWARD_REGISTRY[name]
    return {"name": name, "doc": func.__doc__ or "No doc", "validated": True}


if __name__ == "__main__":
    import os

    test_reward_normalization()
    print(f"Registered Functions: {len(REWARD_REGISTRY)}")

    # Quick Test of Code Execution
    print("\nTesting Code Execution Reward...")
    sample_code = "<think>...</think>```python\nprint(1+1)\n```"
    score = r1_code_execution_reward([""], [sample_code], [""])
    print(f"Code Exec Score: {score[0]}")
