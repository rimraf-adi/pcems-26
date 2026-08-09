import os
# Prevent PyTorch/HuggingFace Tokenizer C++ multithreading segfault on macOS Python 3.13
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re
import time
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter

# Configure Console Logger for Terminal Diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="\033[36m[%(asctime)s]\033[0m \033[1m[%(levelname)s]\033[0m \033[35m[ParaphraseAI]\033[0m %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ParaphraseAI")

# Try importing PyTorch & SentenceTransformers, fallback gracefully to TfidfVectorizer
HAVE_TRANSFORMERS = False
try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# ============================================================================
# EXPANDED MODEL-SPECIFIC FINGERPRINT PHRASE DICTIONARIES
# Each phrase is weighted by empirical frequency differential
# (how much more common it is in AI text vs human academic text)
# Weight scale: 1.0 = mildly AI-associated, 3.0 = strong AI fingerprint
# ============================================================================

GPT_FINGERPRINT_PHRASES = {
    # Strong GPT fingerprints (weight 3.0)
    "delve": 3.0, "tapestry": 3.0, "testament to": 2.5, "it is important to note": 3.0,
    "it's important to note": 3.0, "it's worth noting that": 2.5,
    "in the realm of": 3.0, "beacon of": 2.5, "demystify": 2.5,
    # Moderate GPT fingerprints (weight 2.0)
    "pivotal": 2.0, "overarching": 2.0, "foster": 1.5, "paramount": 2.0,
    "elucidate": 2.0, "multifaceted": 2.0, "underscores": 1.5,
    "paradigm": 1.5, "harness": 1.5, "catalyst": 1.5, "spearhead": 2.0,
    "interplay": 1.5, "intricate": 1.5, "foundational": 1.5,
    "transformative": 2.0, "seamless": 1.5, "synergy": 2.0,
    "holistic": 2.0, "unwavering": 2.0, "embark": 2.0, "cornerstone": 2.0,
    # Mild GPT transition markers (weight 1.0)
    "furthermore": 1.0, "moreover": 1.0, "consequently": 1.0,
    "in conclusion": 1.0, "navigating the": 1.5, "landscape of": 1.5,
    "at the heart of": 1.5, "a myriad of": 2.0, "in today's": 1.5,
    "ever-evolving": 2.5, "game-changer": 2.0, "cutting-edge": 1.5,
}

GEMINI_FINGERPRINT_PHRASES = {
    # Strong Gemini fingerprints
    "shed light on": 3.0, "plays a crucial role": 3.0, "serves as a": 2.0,
    "key takeaway": 3.0, "brings to the fore": 2.5, "paves the way": 2.5,
    "rich array": 2.5, "in light of": 2.0, "vital role": 2.0,
    # Moderate Gemini fingerprints
    "comprehensive": 1.5, "leverage": 1.5, "in summary": 1.5,
    "overall": 1.0, "stands out": 1.5, "noteworthy": 1.5,
    "unravel": 2.0, "highlighting": 1.0, "dynamic": 1.0,
    "it is essential to": 2.0, "a wide range of": 1.5, "it is crucial": 2.0,
    "on the other hand": 1.0, "as a result": 1.0, "to that end": 1.5,
    "a deeper understanding": 2.0, "offers valuable insights": 2.5,
    "worth mentioning": 2.0, "as we can see": 2.0,
}

CLAUDE_FINGERPRINT_PHRASES = {
    # Strong Claude fingerprints
    "it is worth noting": 3.0, "worth highlighting": 2.5,
    "important to consider": 2.5, "nuanced perspective": 3.0,
    "broader context": 2.5, "inextricably": 3.0,
    # Moderate Claude fingerprints
    "nuance": 2.0, "salient": 2.0, "notable": 1.5, "intrinsically": 2.5,
    "alignment": 1.0, "paramount": 1.5, "pivotal": 1.5, "thoughtful": 2.0,
    "balanced": 1.0, "multi-faceted": 2.5,
    "that said": 2.0, "it bears mentioning": 2.5, "with that in mind": 2.0,
    "meaningfully": 2.0, "substantively": 2.0, "a few things": 1.5,
    "to be clear": 2.0, "the key insight": 2.0, "worth unpacking": 2.5,
    "genuinely": 1.5, "straightforward": 1.5,
}

# Common academic transition / discourse marker phrases for Signal 5
DISCOURSE_MARKERS = {
    "furthermore", "moreover", "in addition", "additionally", "however",
    "nevertheless", "nonetheless", "consequently", "therefore", "thus",
    "hence", "accordingly", "meanwhile", "subsequently", "conversely",
    "in contrast", "on the other hand", "as a result", "for instance",
    "for example", "in particular", "specifically", "notably", "indeed",
    "in conclusion", "to summarize", "in summary", "overall",
}

# Passive voice auxiliary patterns for Signal 6
PASSIVE_AUX_PATTERN = re.compile(
    r'\b(is|are|was|were|been|being|be)\s+(\w+ed|built|made|done|seen|given|taken|known|shown|found|said|thought|held|kept|left|met|set|run|put|cut|let|read|told|sold|sent|spent|lent|meant|felt|dealt|led|fed|bred|sped|fled|shed|split|spread|shut|rid|bid|cast|burst|cost|hit|hurt|knit|quit|spit|thrust|wet|wring|clung|flung|stung|swung|wrung|dug|hung|slung|spun|stuck|struck|stunk|strung|sung|sunk|swam|torn|worn|borne|sworn|grown|blown|drawn|flown|frozen|spoken|stolen|chosen|broken|woven|driven|written|risen|ridden|hidden|bitten|beaten|eaten|fallen|forgotten|gotten|shaken|mistaken|overtaken|undertaken|awoken|arisen|forbidden)\b',
    re.IGNORECASE
)

# Hedge phrases for Signal 6
HEDGE_PHRASES = [
    "might", "could potentially", "it seems", "it appears", "arguably",
    "to some extent", "in some cases", "it could be argued", "perhaps",
    "possibly", "presumably", "seemingly", "relatively", "somewhat",
    "tend to", "tends to", "may", "it is possible", "it is likely",
    "it is plausible", "one could argue",
]


def get_device() -> str:
    """Select safe device for sentence embeddings."""
    if HAVE_TRANSFORMERS:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu (TF-IDF Fast Engine)"

def tokenize_words(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric words."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

def split_sentences_raw(text: str) -> List[str]:
    """Split text into non-empty sentences."""
    text = text.strip()
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]

def compute_jaccard_similarity(words1: List[str], words2: List[str]) -> float:
    """Compute lexical Jaccard similarity (exact word token overlap)."""
    set1, set2 = set(words1), set(words2)
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return float(len(intersection) / len(union))

def compute_vocabulary_diversity(words: List[str]) -> float:
    """Compute Type-Token Ratio (TTR) as vocabulary richness index."""
    if not words:
        return 0.0
    return float(len(set(words)) / len(words))

def compute_shannon_entropy(words: List[str]) -> float:
    """Compute Shannon Entropy of unigram distribution as Perplexity proxy."""
    if not words:
        return 0.0
    total_words = len(words)
    counts = Counter(words)
    entropy = -sum((cnt / total_words) * math.log2(cnt / total_words) for cnt in counts.values())
    return round(entropy, 2)


# ============================================================================
# SIGNAL 1: Weighted Fingerprint Phrase Density
# ============================================================================
def compute_fingerprint_density(text: str, phrase_dict: Dict[str, float]) -> Tuple[float, int, float]:
    """
    Compute weighted fingerprint density for a model's phrase dictionary.
    Returns (normalized_score [0-1], raw_match_count, weighted_density_per_1k_words).
    """
    lower_text = text.lower()
    words = tokenize_words(text)
    n_words = max(1, len(words))

    raw_count = 0
    weighted_sum = 0.0
    for phrase, weight in phrase_dict.items():
        # Use word-boundary regex for single words, simple find for multi-word
        if ' ' in phrase:
            matches = len(re.findall(re.escape(phrase), lower_text))
        else:
            matches = len(re.findall(r'\b' + re.escape(phrase) + r'\b', lower_text))
        raw_count += matches
        weighted_sum += matches * weight

    density_per_k = (weighted_sum / n_words) * 1000.0

    # Normalize: 0 density → 0.0 score; 15+ weighted density per 1k → 1.0 score
    normalized = min(1.0, density_per_k / 15.0)
    return round(normalized, 4), raw_count, round(density_per_k, 2)


# ============================================================================
# SIGNAL 2: Sentence Length Uniformity (Burstiness Inverse)
# ============================================================================
def compute_sentence_length_uniformity(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    AI text has uniform sentence lengths (low CV). Human text is bursty (high CV).
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = perfectly uniform (AI-like), 0.0 = highly varied (human-like).
    """
    sents = split_sentences_raw(text)
    if len(sents) < 3:
        return 0.0, {"cv": 0.0, "mean": 0.0, "std": 0.0, "status": "Too few sentences"}

    lengths = [len(tokenize_words(s)) for s in sents]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    cv = std_len / (mean_len + 1e-5)

    # CV < 0.25 → very uniform (AI); CV > 0.65 → very bursty (human)
    # Inverse mapping: lower CV → higher score (more AI-like)
    normalized = max(0.0, min(1.0, 1.0 - (cv / 0.65)))

    if cv >= 0.55:
        status = "High Burstiness (Human-like)"
    elif cv >= 0.35:
        status = "Moderate (Balanced)"
    else:
        status = "Low Burstiness (AI Monotone)"

    return round(normalized, 4), {
        "cv": round(cv, 3),
        "mean": round(mean_len, 1),
        "std": round(std_len, 1),
        "status": status,
    }


# ============================================================================
# SIGNAL 3: Vocabulary Richness Decay Curve
# ============================================================================
def compute_vocab_richness_decay(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Human writers introduce new vocabulary throughout. AI frontloads rare words then recycles.
    Measures hapax legomena ratio and TTR decay across sliding windows.
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = strong vocabulary recycling (AI-like), 0.0 = continuous novelty (human-like).
    """
    words = tokenize_words(text)
    n = len(words)
    if n < 20:
        return 0.0, {"hapax_ratio": 0.0, "ttr_decay": 0.0, "status": "Too short"}

    # Hapax legomena: words appearing exactly once
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = hapax / len(counts) if counts else 0.0

    # TTR decay: Compare TTR of first half vs second half
    mid = n // 2
    first_half = words[:mid]
    second_half = words[mid:]
    ttr_first = len(set(first_half)) / max(1, len(first_half))
    ttr_second = len(set(second_half)) / max(1, len(second_half))

    # If TTR drops significantly in second half, vocabulary is being recycled (AI pattern)
    ttr_decay = max(0.0, ttr_first - ttr_second)

    # AI pattern: low hapax ratio (words get reused) + high TTR decay
    # Human academic: moderate hapax ratio (~0.5), low TTR decay
    hapax_signal = max(0.0, 1.0 - (hapax_ratio / 0.55))  # Lower hapax → higher AI signal
    decay_signal = min(1.0, ttr_decay / 0.15)              # Higher decay → higher AI signal
    normalized = 0.6 * hapax_signal + 0.4 * decay_signal

    if normalized >= 0.65:
        status = "Vocabulary Recycling (AI-like)"
    elif normalized >= 0.35:
        status = "Moderate Richness"
    else:
        status = "Continuous Novelty (Human-like)"

    return round(min(1.0, normalized), 4), {
        "hapax_ratio": round(hapax_ratio, 3),
        "ttr_first_half": round(ttr_first, 3),
        "ttr_second_half": round(ttr_second, 3),
        "ttr_decay": round(ttr_decay, 3),
        "status": status,
    }


# ============================================================================
# SIGNAL 4: Sentence Opener Diversity
# ============================================================================
def compute_opener_diversity(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    AI models reuse sentence-initial patterns. Measures entropy of first-2-word bigrams.
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = repetitive openers (AI-like), 0.0 = diverse openers (human-like).
    """
    sents = split_sentences_raw(text)
    if len(sents) < 4:
        return 0.0, {"opener_entropy": 0.0, "unique_ratio": 1.0, "status": "Too few sentences"}

    # Extract first-2-word opener bigrams
    openers = []
    for s in sents:
        w = tokenize_words(s)
        if len(w) >= 2:
            openers.append(f"{w[0]} {w[1]}")
        elif len(w) == 1:
            openers.append(w[0])

    if not openers:
        return 0.0, {"opener_entropy": 0.0, "unique_ratio": 1.0, "status": "No openers"}

    n_openers = len(openers)
    unique_openers = len(set(openers))
    unique_ratio = unique_openers / n_openers

    # Shannon entropy of opener distribution
    opener_counts = Counter(openers)
    entropy = -sum((c / n_openers) * math.log2(c / n_openers) for c in opener_counts.values())

    # Max possible entropy (all unique)
    max_entropy = math.log2(n_openers) if n_openers > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Low entropy + low unique ratio → AI-like (repetitive)
    normalized = max(0.0, min(1.0, 1.0 - normalized_entropy))

    if normalized >= 0.60:
        status = "Repetitive Openers (AI Pattern)"
    elif normalized >= 0.30:
        status = "Moderate Diversity"
    else:
        status = "Diverse Openers (Human-like)"

    return round(normalized, 4), {
        "opener_entropy": round(entropy, 3),
        "max_entropy": round(max_entropy, 3),
        "unique_ratio": round(unique_ratio, 3),
        "top_openers": dict(opener_counts.most_common(5)),
        "status": status,
    }


# ============================================================================
# SIGNAL 5: Transition Phrase Regularity
# ============================================================================
def compute_transition_regularity(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    AI inserts discourse markers at predictable intervals. Measures CV of inter-marker distances.
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = metronomically regular transitions (AI-like), 0.0 = irregular (human-like).
    """
    sents = split_sentences_raw(text)
    if len(sents) < 5:
        return 0.0, {"marker_count": 0, "interval_cv": 0.0, "status": "Too few sentences"}

    lower_text = text.lower()

    # Find positions (word indices) of each discourse marker occurrence
    marker_positions = []
    words = tokenize_words(text)
    for i, w in enumerate(words):
        # Check single-word markers
        if w in DISCOURSE_MARKERS:
            marker_positions.append(i)

    # Also check multi-word markers
    for marker in DISCOURSE_MARKERS:
        if ' ' in marker:
            for m in re.finditer(re.escape(marker), lower_text):
                # Approximate word position from character position
                approx_word_pos = len(tokenize_words(lower_text[:m.start()]))
                marker_positions.append(approx_word_pos)

    marker_positions = sorted(set(marker_positions))
    n_markers = len(marker_positions)

    if n_markers < 3:
        return 0.0, {"marker_count": n_markers, "interval_cv": 0.0, "status": "Insufficient markers"}

    # Compute inter-marker distances
    intervals = [marker_positions[i+1] - marker_positions[i] for i in range(n_markers - 1)]
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    interval_cv = std_interval / (mean_interval + 1e-5)

    # Low CV = regular spacing (AI-like); High CV = irregular (human-like)
    # CV < 0.3 → very regular; CV > 1.0 → very irregular
    normalized = max(0.0, min(1.0, 1.0 - (interval_cv / 1.2)))

    # Also factor in marker density (AI over-uses transitions)
    n_words = max(1, len(words))
    marker_density = (n_markers / n_words) * 100.0  # markers per 100 words
    density_boost = min(0.2, max(0.0, (marker_density - 3.0) * 0.05))  # boost if > 3 per 100 words

    normalized = min(1.0, normalized + density_boost)

    if normalized >= 0.60:
        status = "Metronomic Transitions (AI Pattern)"
    elif normalized >= 0.30:
        status = "Moderate Regularity"
    else:
        status = "Irregular Transitions (Human-like)"

    return round(normalized, 4), {
        "marker_count": n_markers,
        "mean_interval_words": round(mean_interval, 1),
        "interval_cv": round(interval_cv, 3),
        "marker_density_per_100w": round(marker_density, 2),
        "status": status,
    }


# ============================================================================
# SIGNAL 6: Passive Voice & Hedging Ratio
# ============================================================================
def compute_passive_hedge_ratio(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Different AI models have distinct passive/hedge profiles.
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = high hedge/passive density (AI-like), 0.0 = direct/active (human-like).
    """
    sents = split_sentences_raw(text)
    words = tokenize_words(text)
    n_words = max(1, len(words))
    n_sents = max(1, len(sents))

    # Count passive voice constructions
    passive_count = len(PASSIVE_AUX_PATTERN.findall(text))

    # Count hedge phrases
    lower_text = text.lower()
    hedge_count = 0
    for phrase in HEDGE_PHRASES:
        if ' ' in phrase:
            hedge_count += len(re.findall(re.escape(phrase), lower_text))
        else:
            hedge_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', lower_text))

    # Passive density per sentence
    passive_per_sent = passive_count / n_sents
    # Hedge density per 1k words
    hedge_per_k = (hedge_count / n_words) * 1000.0

    # Normalized: AI text tends to have 0.3-0.5 passives/sent and 8+ hedges per 1k
    passive_signal = min(1.0, passive_per_sent / 0.5)
    hedge_signal = min(1.0, hedge_per_k / 12.0)
    normalized = 0.45 * passive_signal + 0.55 * hedge_signal

    if normalized >= 0.55:
        status = "High Hedging/Passive (AI Caution)"
    elif normalized >= 0.25:
        status = "Moderate Voice"
    else:
        status = "Direct/Active (Human Confident)"

    return round(min(1.0, normalized), 4), {
        "passive_count": passive_count,
        "passive_per_sent": round(passive_per_sent, 3),
        "hedge_count": hedge_count,
        "hedge_per_1k": round(hedge_per_k, 1),
        "status": status,
    }


# ============================================================================
# SIGNAL 7: Zipf's Law Deviation
# ============================================================================
def compute_zipf_deviation(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Natural language follows Zipf's distribution (slope ≈ -1.0 on log-log plot).
    AI text produces smoother, less Zipfian distributions.
    Returns (normalized_score [0-1], detail_dict).
    Score 1.0 = strongly non-Zipfian (AI-like), 0.0 = natural Zipf (human-like).
    """
    words = tokenize_words(text)
    if len(words) < 30:
        return 0.0, {"zipf_slope": -1.0, "r_squared": 1.0, "status": "Too short"}

    counts = Counter(words)
    freqs = sorted(counts.values(), reverse=True)
    n_unique = len(freqs)
    if n_unique < 5:
        return 0.0, {"zipf_slope": -1.0, "r_squared": 1.0, "status": "Too few unique words"}

    # Compute log-log linear regression: log(freq) = slope * log(rank) + intercept
    log_ranks = np.log(np.arange(1, n_unique + 1, dtype=float))
    log_freqs = np.log(np.array(freqs, dtype=float))

    # Simple linear regression
    mean_x = np.mean(log_ranks)
    mean_y = np.mean(log_freqs)
    ss_xy = np.sum((log_ranks - mean_x) * (log_freqs - mean_y))
    ss_xx = np.sum((log_ranks - mean_x) ** 2)
    slope = ss_xy / (ss_xx + 1e-10)

    # R² (goodness of fit)
    predicted = slope * (log_ranks - mean_x) + mean_y
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - mean_y) ** 2)
    r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))

    # Natural text: slope ≈ -1.0, R² ≈ 0.85-0.95
    # AI text: slope ≈ -0.6 to -0.8, R² ≈ 0.7-0.85 (flatter, more uniform word frequency)
    slope_deviation = abs(slope - (-1.0))  # Deviation from ideal Zipf slope
    # High deviation from -1.0 → AI-like; Low deviation → human Zipfian
    slope_signal = min(1.0, slope_deviation / 0.5)

    # Low R² can also indicate AI (less clean Zipf fit)
    fit_signal = max(0.0, min(1.0, 1.0 - r_squared))

    normalized = 0.7 * slope_signal + 0.3 * fit_signal

    if normalized >= 0.55:
        status = "Non-Zipfian (AI Distribution)"
    elif normalized >= 0.25:
        status = "Moderate Zipf Adherence"
    else:
        status = "Natural Zipfian (Human-like)"

    return round(min(1.0, normalized), 4), {
        "zipf_slope": round(float(slope), 3),
        "r_squared": round(float(max(0.0, r_squared)), 3),
        "slope_deviation": round(float(slope_deviation), 3),
        "n_unique_words": n_unique,
        "status": status,
    }


# ============================================================================
# COMPOSITE AI SIGNAL PROFILE: Orchestrates all 7 signals per model
# ============================================================================

# Model-specific weight profiles (research-informed behavioral differences)
MODEL_SIGNAL_WEIGHTS = {
    "gpt": {
        "fingerprint":   0.20,
        "uniformity":    0.15,
        "vocab_decay":   0.15,
        "opener":        0.15,
        "transition":    0.10,
        "passive_hedge": 0.10,
        "zipf":          0.15,
    },
    "gemini": {
        "fingerprint":   0.15,
        "uniformity":    0.20,
        "vocab_decay":   0.15,
        "opener":        0.10,
        "transition":    0.15,
        "passive_hedge": 0.10,
        "zipf":          0.15,
    },
    "claude": {
        "fingerprint":   0.15,
        "uniformity":    0.10,
        "vocab_decay":   0.15,
        "opener":        0.15,
        "transition":    0.15,
        "passive_hedge": 0.20,
        "zipf":          0.10,
    },
}

def compute_ai_signal_profile(text: str) -> Dict[str, Any]:
    """
    Compute all 7 AI detection signals and produce separate per-model risk indices.
    Returns a comprehensive profile with individual signal scores and per-model composites.
    """
    # Signal 1: Fingerprint Densities (per-model)
    gpt_fp_score, gpt_fp_count, gpt_fp_density = compute_fingerprint_density(text, GPT_FINGERPRINT_PHRASES)
    gem_fp_score, gem_fp_count, gem_fp_density = compute_fingerprint_density(text, GEMINI_FINGERPRINT_PHRASES)
    cla_fp_score, cla_fp_count, cla_fp_density = compute_fingerprint_density(text, CLAUDE_FINGERPRINT_PHRASES)

    # Signal 2: Sentence Length Uniformity (shared across models)
    uniformity_score, uniformity_detail = compute_sentence_length_uniformity(text)

    # Signal 3: Vocabulary Richness Decay (shared)
    vocab_score, vocab_detail = compute_vocab_richness_decay(text)

    # Signal 4: Sentence Opener Diversity (shared)
    opener_score, opener_detail = compute_opener_diversity(text)

    # Signal 5: Transition Regularity (shared)
    transition_score, transition_detail = compute_transition_regularity(text)

    # Signal 6: Passive/Hedge Ratio (shared)
    passive_score, passive_detail = compute_passive_hedge_ratio(text)

    # Signal 7: Zipf Deviation (shared)
    zipf_score, zipf_detail = compute_zipf_deviation(text)

    # Compose per-model risk indices using model-specific weights
    def _composite(model_key: str, fp_score: float) -> float:
        w = MODEL_SIGNAL_WEIGHTS[model_key]
        score = (
            w["fingerprint"]   * fp_score +
            w["uniformity"]    * uniformity_score +
            w["vocab_decay"]   * vocab_score +
            w["opener"]        * opener_score +
            w["transition"]    * transition_score +
            w["passive_hedge"] * passive_score +
            w["zipf"]          * zipf_score
        )
        return round(min(100.0, score * 100.0), 1)

    gpt_risk_pct = _composite("gpt", gpt_fp_score)
    gemini_risk_pct = _composite("gemini", gem_fp_score)
    claude_risk_pct = _composite("claude", cla_fp_score)

    # Per-model signal breakdowns for radar charts
    def _signal_vector(model_key: str, fp_score: float) -> Dict[str, float]:
        return {
            "Fingerprint Density": round(fp_score, 3),
            "Sentence Uniformity": round(uniformity_score, 3),
            "Vocab Richness Decay": round(vocab_score, 3),
            "Opener Diversity": round(opener_score, 3),
            "Transition Regularity": round(transition_score, 3),
            "Passive/Hedge Ratio": round(passive_score, 3),
            "Zipf Deviation": round(zipf_score, 3),
        }

    # Legacy burstiness / perplexity fields for backward compatibility
    words = tokenize_words(text)
    sents = split_sentences_raw(text)
    entropy = compute_shannon_entropy(words)

    burstiness_cv = uniformity_detail.get("cv", 0.0)
    if burstiness_cv >= 0.55:
        burst_status = "High (Dynamic Human Rhythm)"
        burst_color = "green"
    elif burstiness_cv >= 0.35:
        burst_status = "Moderate (Balanced Cadence)"
        burst_color = "orange"
    else:
        burst_status = "Low (Uniform AI Monotone)"
        burst_color = "red"

    if entropy >= 7.0:
        perp_status = "High Perplexity (Unpredictable / Human-like)"
    elif entropy >= 5.5:
        perp_status = "Moderate Perplexity (Standard Academic)"
    else:
        perp_status = "Low Perplexity (Predictable / Formulaic)"

    return {
        # Per-model composite risk percentages (separate, unmerged)
        "gpt_risk_pct": gpt_risk_pct,
        "gemini_risk_pct": gemini_risk_pct,
        "claude_risk_pct": claude_risk_pct,

        # Per-model signal vectors (for radar charts)
        "gpt_signals": _signal_vector("gpt", gpt_fp_score),
        "gemini_signals": _signal_vector("gemini", gem_fp_score),
        "claude_signals": _signal_vector("claude", cla_fp_score),

        # Per-model fingerprint details
        "gpt_fp_count": gpt_fp_count,
        "gpt_fp_density": gpt_fp_density,
        "gemini_fp_count": gem_fp_count,
        "gemini_fp_density": gem_fp_density,
        "claude_fp_count": cla_fp_count,
        "claude_fp_density": cla_fp_density,

        # Individual signal detail dictionaries (for expandable UI panels)
        "signal_details": {
            "uniformity": uniformity_detail,
            "vocab_decay": vocab_detail,
            "opener_diversity": opener_detail,
            "transition_regularity": transition_detail,
            "passive_hedge": passive_detail,
            "zipf_deviation": zipf_detail,
        },

        # Legacy backward-compatible fields
        "burstiness_score": round(burstiness_cv, 3),
        "burstiness_status": burst_status,
        "burstiness_color": burst_color,
        "perplexity_entropy": entropy,
        "perplexity_status": perp_status,
        "gpt_count": gpt_fp_count,
        "gemini_count": gem_fp_count,
        "claude_count": cla_fp_count,
        "gpt_density_per_k": gpt_fp_density,
        "gemini_density_per_k": gem_fp_density,
        "claude_density_per_k": cla_fp_density,
        "sentence_len_mean": uniformity_detail.get("mean", 0.0),
        "sentence_len_std": uniformity_detail.get("std", 0.0),
    }

def compute_readability_metrics(text: str) -> Dict[str, float]:
    """Compute basic readability and prose metrics."""
    words = tokenize_words(text)
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    num_words = len(words)
    num_sents = max(1, len(sents))
    
    avg_sent_len = num_words / num_sents
    avg_word_len = sum(len(w) for w in words) / max(1, num_words)
    
    # Approximate Flesch Reading Ease score
    syllables = sum(max(1, len(re.findall(r'[aeiouyAEIOUY]', w))) for w in words)
    flesch_score = 206.835 - (1.015 * avg_sent_len) - (84.6 * (syllables / max(1, num_words)))
    
    return {
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_word_length": round(avg_word_len, 2),
        "readability_score": round(max(0.0, min(100.0, flesch_score)), 1),
    }


class ParaphraseAnalyzer:
    """Enhanced multi-dimensional manuscript similarity and paraphrase quality engine."""

    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2 (Fast & Lightweight ~80MB)": "all-MiniLM-L6-v2",
        "all-mpnet-base-v2 (High Accuracy ~400MB)": "all-mpnet-base-v2",
        "TF-IDF Fast Offline Mode (0 Downloads Required)": "tfidf-fast",
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        start_t = time.time()
        self.model_name = model_name
        self.device = get_device()
        self._cache = {}
        self.use_transformers = False

        if HAVE_TRANSFORMERS and model_name != "tfidf-fast":
            try:
                logger.info(f"Loading Neural Model: \033[33m{model_name}\033[0m on \033[32m{self.device.upper()}\033[0m")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.use_transformers = True
                logger.info(f"Neural Model ready in {time.time() - start_t:.2f}s")
            except Exception as e:
                logger.warning(f"Neural model load skipped ({e}). Falling back to fast TF-IDF engine.")
                self.use_transformers = False
        else:
            logger.info("⚡ Operating in Fast Offline Mode (TF-IDF Cosine Similarity)")
            self.use_transformers = False

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into non-empty paragraphs."""
        raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text)]
        return [p for p in raw_paras if p]

    def split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitter."""
        text = text.strip()
        if not text:
            return []
        raw_sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sents if s.strip()]

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract high-dimensional vector embeddings for a list of texts."""
        if not texts:
            return np.zeros((0, 384))
        if self.use_transformers:
            embeddings = self.model.encode(texts, convert_to_numpy=True, device=self.device)
            return embeddings
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            vecs = vectorizer.fit_transform(texts).toarray()
            return vecs

    def compute_similarity_matrix(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix between two text lists."""
        if not texts1 or not texts2:
            return np.zeros((len(texts1), len(texts2)))

        if self.use_transformers:
            t0 = time.time()
            e1 = self.model.encode(texts1, convert_to_tensor=True, device=self.device)
            e2 = self.model.encode(texts2, convert_to_tensor=True, device=self.device)
            sim_tensor = util.cos_sim(e1, e2)
            matrix = sim_tensor.cpu().numpy()
            return matrix
        else:
            all_texts = texts1 + texts2
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_texts)
            v1 = vectorizer.transform(texts1)
            v2 = vectorizer.transform(texts2)
            return cosine_similarity(v1, v2)

    def align_sentences(self, orig_sents: List[str], para_sents: List[str], sentence_matrix: np.ndarray) -> List[int]:
        """
        Compute 1-to-1 positional and semantic sentence alignment indices.
        Returns a list of best_match_idx for each orig_sent preserving 1-to-1 sequential integrity.
        """
        n1 = len(orig_sents)
        n2 = len(para_sents)
        if n1 == 0 or n2 == 0:
            return []

        # Use a hybrid score balancing vector similarity and relative positional alignment
        # to handle cases where sentence counts are equal but content has been rearranged
        aligned_indices = []
        for i in range(n1):
            target_pos = (i / max(1, n1 - 1)) * (n2 - 1) if n1 > 1 else 0
            best_j = 0
            best_score = -1.0
            
            for j in range(n2):
                pos_distance = abs(j - target_pos) / max(1, n2)
                pos_score = max(0.0, 1.0 - pos_distance)
                # Hybrid score balancing vector similarity and relative positional alignment
                combined_score = 0.6 * sentence_matrix[i, j] + 0.4 * pos_score
                if combined_score > best_score:
                    best_score = combined_score
                    best_j = j
            aligned_indices.append(best_j)
        return aligned_indices

    def compute_pca_embedding_space(self, orig_sents: List[str], para_sents: List[str]) -> Dict[str, Any]:
        """Reduce high-dimensional sentence embeddings to 2D & 3D space using PCA for visual vector graph."""
        if not orig_sents or not para_sents:
            return {"nodes_2d": [], "nodes_3d": [], "trajectories": []}

        # Combine all sentences
        all_sents = orig_sents + para_sents
        n1 = len(orig_sents)
        n2 = len(para_sents)

        # 1. Get raw embedding vectors
        if self.use_transformers:
            raw_vecs = self.model.encode(all_sents, convert_to_numpy=True, device=self.device)
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            raw_vecs = vectorizer.fit_transform(all_sents).toarray()

        # 2. PCA 2D Reduction
        n_comp_2d = min(2, raw_vecs.shape[0], raw_vecs.shape[1])
        pca_2d = PCA(n_components=n_comp_2d)
        coords_2d = pca_2d.fit_transform(raw_vecs)
        if coords_2d.shape[1] < 2:
            coords_2d = np.pad(coords_2d, ((0,0), (0, 2 - coords_2d.shape[1])))

        # 3. PCA 3D Reduction
        n_comp_3d = min(3, raw_vecs.shape[0], raw_vecs.shape[1])
        pca_3d = PCA(n_components=n_comp_3d)
        coords_3d = pca_3d.fit_transform(raw_vecs)
        if coords_3d.shape[1] < 3:
            coords_3d = np.pad(coords_3d, ((0,0), (0, 3 - coords_3d.shape[1])))

        # 4. Build Nodes
        nodes_df = []
        for idx, text in enumerate(all_sents):
            is_orig = (idx < n1)
            local_id = (idx + 1) if is_orig else (idx - n1 + 1)
            doc_label = "Baseline content.tex" if is_orig else "Current content.tex"
            
            nodes_df.append({
                "id": idx,
                "local_id": local_id,
                "doc": doc_label,
                "text": text,
                "text_short": text[:60] + "..." if len(text) > 60 else text,
                "x2d": float(coords_2d[idx, 0]),
                "y2d": float(coords_2d[idx, 1]),
                "x3d": float(coords_3d[idx, 0]),
                "y3d": float(coords_3d[idx, 1]),
                "z3d": float(coords_3d[idx, 2]),
            })

        # 5. Build 1-to-1 Trajectories using positional alignment
        sim_matrix = self.compute_similarity_matrix(orig_sents, para_sents)
        aligned_pairs = self.align_sentences(orig_sents, para_sents, sim_matrix)
        trajectories = []
        for i in range(n1):
            best_j = aligned_pairs[i] if i < len(aligned_pairs) else 0
            best_sim = float(sim_matrix[i, best_j])
            
            # 2D Euclidean distance in projected PCA space
            dx2d = coords_2d[i, 0] - coords_2d[n1 + best_j, 0]
            dy2d = coords_2d[i, 1] - coords_2d[n1 + best_j, 1]
            pca_dist = float(np.sqrt(dx2d**2 + dy2d**2))

            trajectories.append({
                "orig_id": i + 1,
                "match_id": best_j + 1,
                "orig_x2d": float(coords_2d[i, 0]),
                "orig_y2d": float(coords_2d[i, 1]),
                "match_x2d": float(coords_2d[n1 + best_j, 0]),
                "match_y2d": float(coords_2d[n1 + best_j, 1]),
                "orig_x3d": float(coords_3d[i, 0]),
                "orig_y3d": float(coords_3d[i, 1]),
                "orig_z3d": float(coords_3d[i, 2]),
                "match_x3d": float(coords_3d[n1 + best_j, 0]),
                "match_y3d": float(coords_3d[n1 + best_j, 1]),
                "match_z3d": float(coords_3d[n1 + best_j, 2]),
                "similarity": round(best_sim, 4),
                "pca_distance": round(pca_dist, 4),
                "orig_text": orig_sents[i],
                "match_text": para_sents[best_j],
            })

        return {
            "nodes": pd.DataFrame(nodes_df),
            "trajectories": pd.DataFrame(trajectories),
            "pca_2d_explained_var": [round(float(v)*100, 1) for v in pca_2d.explained_variance_ratio_] if hasattr(pca_2d, "explained_variance_ratio_") else [0, 0],
            "pca_3d_explained_var": [round(float(v)*100, 1) for v in pca_3d.explained_variance_ratio_] if hasattr(pca_3d, "explained_variance_ratio_") else [0, 0, 0]
        }

    def analyze(
        self, original_text: str, paraphrased_text: str, high_similarity_threshold: float = 0.10
    ) -> Dict[str, Any]:
        """Run complete multi-dimensional manuscript analysis with Perplexity, Burstiness & Embedding Space Graph."""
        start_time = time.time()
        orig_text = original_text.strip()
        para_text = paraphrased_text.strip()

        if not orig_text or not para_text:
            return {"error": "Both original and paraphrased texts must be non-empty."}

        words_orig = tokenize_words(orig_text)
        words_para = tokenize_words(para_text)

        # 1. Multi-Dimensional Metrics Computation
        jaccard_sim = compute_jaccard_similarity(words_orig, words_para)
        ttr_orig = compute_vocabulary_diversity(words_orig)
        ttr_para = compute_vocabulary_diversity(words_para)
        
        read_orig = compute_readability_metrics(orig_text)
        read_para = compute_readability_metrics(para_text)

        # Compute AI Signal Forensics
        bp_orig = compute_ai_signal_profile(orig_text)
        bp_para = compute_ai_signal_profile(para_text)

        # Document Level Semantic Similarity
        doc_matrix = self.compute_similarity_matrix([orig_text], [para_text])
        semantic_sim = float(np.clip(doc_matrix[0, 0], 0.0, 1.0))

        # Lexical Replacement Rate (1 - Jaccard overlap)
        lexical_replacement_rate = 1.0 - jaccard_sim

        # SEPARATE, UNMERGED INDIVIDUAL DETECTOR INDICES
        # 1. Pure Lexical Plagiarism Overlap Index (%)
        turnitin_plagiarism_pct = round(jaccard_sim * 100.0, 1)

        # Sentence Cadence / Monotone AI Rhythm Penalty
        burst_cv = bp_para.get("burstiness_score", 0.4)
        burst_penalty = max(0.0, (0.45 - burst_cv) * 20.0) if burst_cv < 0.45 else 0.0

        # Incorporate lexical overlap with the AI specific signal risk
        gpt_risk_pct = round(min(100.0, (jaccard_sim * 35.0) + bp_para.get("gpt_risk_pct", 0.0) + burst_penalty), 1)
        gemini_risk_pct = round(min(100.0, (jaccard_sim * 35.0) + bp_para.get("gemini_risk_pct", 0.0) + burst_penalty), 1)
        claude_risk_pct = round(min(100.0, (jaccard_sim * 35.0) + bp_para.get("claude_risk_pct", 0.0) + burst_penalty), 1)

        # Highest risk vector among all detectors
        max_detector_risk = max(turnitin_plagiarism_pct, gpt_risk_pct, gemini_risk_pct, claude_risk_pct)
        turnitin_compliant = bool(max_detector_risk <= 10.0)

        # Status & Color Evaluation
        if max_detector_risk > 25.0:
            status_title = f"High Risk Flag: Max Vector at {max_detector_risk}%"
            status_desc = f"Specific risk vectors exceed submission limits (Plagiarism: {turnitin_plagiarism_pct}%, GPT: {gpt_risk_pct}%, Gemini: {gemini_risk_pct}%, Claude: {claude_risk_pct}%). Target individual high-risk cards below."
            status_color = "red"
            status_badge = "🔴 Exceeds 10% Cutoff"
        elif max_detector_risk > 10.0:
            status_title = f"Moderate Risk: Max Vector at {max_detector_risk}%"
            status_desc = f"Near target! (Plagiarism: {turnitin_plagiarism_pct}%, GPT: {gpt_risk_pct}%, Gemini: {gemini_risk_pct}%, Claude: {claude_risk_pct}%). Reduce specific model fingerprint terms."
            status_color = "orange"
            status_badge = "🟡 Near 10% Cutoff"
        else:
            status_title = "Turnitin & Multi-AI Compliant (<10% Target Met Across All Vectors)"
            status_desc = f"All individual detector vectors are under 10%! (Plagiarism: {turnitin_plagiarism_pct}%, GPT: {gpt_risk_pct}%, Gemini: {gemini_risk_pct}%, Claude: {claude_risk_pct}%). Preserves {int(semantic_sim*100)}% semantic fidelity."
            status_color = "green"
            status_badge = "🟢 Compliant (<10%)"

        # 2. Paragraph Level Breakdown (1-to-1 Aligned)
        orig_paras = self.split_paragraphs(orig_text)
        para_paras = self.split_paragraphs(para_text)

        paragraph_analysis = []
        para_matrix = np.zeros((len(orig_paras), len(para_paras)))

        if orig_paras and para_paras:
            para_matrix = self.compute_similarity_matrix(orig_paras, para_paras)
            n_p1 = len(orig_paras)
            n_p2 = len(para_paras)
            for i, p_orig in enumerate(orig_paras):
                if n_p1 == n_p2:
                    best_idx = i
                else:
                    best_idx = int(np.argmax(para_matrix[i]))
                
                best_sim = float(para_matrix[i, best_idx]) if best_idx < n_p2 else 0.0
                best_para_match = para_paras[best_idx] if best_idx < n_p2 else ""
                
                p_words_orig = tokenize_words(p_orig)
                p_words_para = tokenize_words(best_para_match)
                p_jaccard = compute_jaccard_similarity(p_words_orig, p_words_para)

                paragraph_analysis.append({
                    "orig_index": i + 1,
                    "orig_paragraph": p_orig,
                    "best_match_index": best_idx + 1,
                    "best_match_paragraph": best_para_match,
                    "semantic_similarity": round(best_sim, 4),
                    "lexical_overlap": round(p_jaccard, 4),
                    "word_count_orig": len(p_orig.split()),
                    "word_count_para": len(best_para_match.split()),
                })

        # 3. Sentence Level Breakdown & Turnitin Risk Assessment (1-to-1 Positional Matching)
        orig_sents = self.split_sentences(orig_text)
        para_sents = self.split_sentences(para_text)

        sentence_analysis = []
        sentence_matrix = np.zeros((len(orig_sents), len(para_sents)))

        verbatim_copy_count = 0
        high_risk_count = 0
        moderate_risk_count = 0
        low_risk_count = 0

        if orig_sents and para_sents:
            sentence_matrix = self.compute_similarity_matrix(orig_sents, para_sents)
            aligned_sent_indices = self.align_sentences(orig_sents, para_sents, sentence_matrix)

            for i, s_orig in enumerate(orig_sents):
                best_idx = aligned_sent_indices[i] if i < len(aligned_sent_indices) else 0
                best_sim = float(sentence_matrix[i, best_idx])
                best_sent_match = para_sents[best_idx] if best_idx < len(para_sents) else ""

                s_w1 = tokenize_words(s_orig)
                s_w2 = tokenize_words(best_sent_match)
                s_lexical_sim = compute_jaccard_similarity(s_w1, s_w2)

                if best_sim >= 0.95 or s_lexical_sim >= 0.85:
                    risk_level = "Direct Verbatim Copy (Turnitin High Risk)"
                    risk_badge = "🔴 Verbatim (High Risk)"
                    verbatim_copy_count += 1
                    high_risk_count += 1
                elif s_lexical_sim >= high_similarity_threshold or best_sim >= 0.80:
                    risk_level = "High Similarity (Turnitin Risk >10%)"
                    risk_badge = "🟠 Exceeds 10% Risk"
                    high_risk_count += 1
                elif s_lexical_sim >= 0.05:
                    risk_level = "Moderate Rephrase (Near <10% Target)"
                    risk_badge = "🟡 Moderate (5-10%)"
                    moderate_risk_count += 1
                else:
                    risk_level = "Turnitin Safe (<10% Similarity)"
                    risk_badge = "🟢 Turnitin Safe (<10%)"
                    low_risk_count += 1

                s_lower = best_sent_match.lower()
                gpt_flags = [p for p in GPT_FINGERPRINT_PHRASES if p in s_lower]
                gemini_flags = [p for p in GEMINI_FINGERPRINT_PHRASES if p in s_lower]
                claude_flags = [p for p in CLAUDE_FINGERPRINT_PHRASES if p in s_lower]
                
                ai_flags = []
                if gpt_flags: ai_flags.append("ChatGPT")
                if gemini_flags: ai_flags.append("Gemini")
                if claude_flags: ai_flags.append("Claude")

                sentence_analysis.append({
                    "orig_id": i + 1,
                    "orig_sentence": s_orig,
                    "best_match_id": best_idx + 1,
                    "best_match_sentence": best_sent_match,
                    "similarity": round(best_sim, 4),
                    "lexical_overlap": round(s_lexical_sim, 4),
                    "risk_level": risk_level,
                    "risk_badge": risk_badge,
                    "ai_flags": ", ".join(ai_flags) if ai_flags else "None",
                    "flagged_phrases": ", ".join(set(gpt_flags + gemini_flags + claude_flags))
                })

        # 4. Compute PCA Vector Embedding Graph Projections
        pca_embedding_data = self.compute_pca_embedding_space(orig_sents, para_sents)

        all_sent_sims = [s["similarity"] for s in sentence_analysis] if sentence_analysis else [0.0]

        total_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ Multi-Dimensional Evaluation Complete in {total_ms:.1f}ms")

        return {
            "device": self.device,
            "engine": "Neural CPU" if self.use_transformers else "Fast TF-IDF",
            "model_name": self.model_name,
            "document": {
                "semantic_similarity": round(semantic_sim, 4),
                "jaccard_similarity": round(jaccard_sim, 4),
                "lexical_replacement_rate": round(lexical_replacement_rate, 4),
                "turnitin_plagiarism_pct": turnitin_plagiarism_pct,
                "gpt_risk_pct": gpt_risk_pct,
                "gemini_risk_pct": gemini_risk_pct,
                "claude_risk_pct": claude_risk_pct,
                "max_detector_risk": max_detector_risk,
                "turnitin_risk_pct": max_detector_risk,
                "turnitin_compliant": turnitin_compliant,
                "status_title": status_title,
                "status_desc": status_desc,
                "status_color": status_color,
                "status_badge": status_badge,
                "orig_word_count": len(words_orig),
                "para_word_count": len(words_para),
                "orig_vocab_richness": round(ttr_orig, 4),
                "para_vocab_richness": round(ttr_para, 4),
                "readability_baseline": read_orig,
                "readability_current": read_para,
                "burstiness_perplexity_baseline": bp_orig,
                "burstiness_perplexity_current": bp_para,
            },
            "paragraph": {
                "orig_count": len(orig_paras),
                "para_count": len(para_paras),
                "matches": paragraph_analysis,
                "matrix": para_matrix,
            },
            "sentence": {
                "orig_count": len(orig_sents),
                "para_count": len(para_sents),
                "matches": sentence_analysis,
                "matrix": sentence_matrix,
                "metrics": {
                    "mean_similarity": round(float(np.mean(all_sent_sims)), 4),
                    "max_similarity": round(float(np.max(all_sent_sims)), 4),
                    "min_similarity": round(float(np.min(all_sent_sims)), 4),
                    "verbatim_copy_count": verbatim_copy_count,
                    "high_risk_count": high_risk_count,
                    "moderate_risk_count": moderate_risk_count,
                    "low_risk_count": low_risk_count,
                    "turnitin_risk_pct": max_detector_risk,
                }
            },
            "embeddings_pca": pca_embedding_data
        }


if __name__ == "__main__":
    orig = "The machine learning model was trained on thousands of medical images to identify rare diseases early."
    para = "Thousands of clinical images were utilized to train the AI system for detecting uncommon illnesses in early stages."
    
    analyzer = ParaphraseAnalyzer("all-MiniLM-L6-v2")
    results = analyzer.analyze(orig, para)
    print("\n--- Multi-Dimensional Results ---")
    print(f"Engine: {results['engine']}")
    print(f"Semantic Sim: {results['document']['semantic_similarity']*100:.2f}%")
    print(f"PCA Nodes Count: {len(results['embeddings_pca']['nodes'])}")
