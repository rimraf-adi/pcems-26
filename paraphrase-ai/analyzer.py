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

# Expanded AI transition & signature GPT vocabulary terms
AI_TRANSITION_WORDS = {
    "therefore", "however", "furthermore", "moreover", "consequently",
    "in conclusion", "it is important to note", "additionally", "thus",
    "in summary", "overall", "nonetheless", "hence", "importantly",
    "delve", "tapestry", "testament", "pivotal", "overarching", "foster",
    "beacon", "paramount", "elucidate", "multifaceted", "underscores",
    "paradigm", "realm", "harness", "catalyst", "spearhead", "interplay",
    "intricate", "foundational", "transformative", "seamless", "synergy",
    "holistic", "unwavering", "demystify", "embark", "cornerstone",
    "it is worth noting", "serves as a", "shed light on", "plays a crucial role"
}

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

def compute_burstiness_and_perplexity(text: str) -> Dict[str, Any]:
    """
    Compute Burstiness (variance in sentence length/rhythm) 
    and Perplexity proxy (word choice unpredictability & AI transition frequency).
    """
    words = tokenize_words(text)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sents:
        return {
            "burstiness_score": 0.0,
            "burstiness_status": "Uniform",
            "perplexity_entropy": 0.0,
            "perplexity_status": "Low Predictability",
            "ai_transition_count": 0,
            "ai_density_per_k": 0.0,
            "sentence_len_std": 0.0,
        }

    sent_lengths = [len(tokenize_words(s)) for s in sents]
    mean_len = np.mean(sent_lengths) if sent_lengths else 0.0
    std_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0

    # Burstiness Coefficient CV = std / (mean + 1e-5)
    burstiness_cv = float(std_len / (mean_len + 1e-5))

    if burstiness_cv >= 0.55:
        burstiness_status = "High (Dynamic Human Rhythm)"
        burstiness_color = "green"
    elif burstiness_cv >= 0.35:
        burstiness_status = "Moderate (Balanced Cadence)"
        burstiness_color = "orange"
    else:
        burstiness_status = "Low (Uniform AI Monotone)"
        burstiness_color = "red"

    # Perplexity Proxy via Shannon Entropy
    entropy = compute_shannon_entropy(words)
    if entropy >= 7.0:
        perplexity_status = "High Perplexity (Unpredictable / Human-like)"
    elif entropy >= 5.5:
        perplexity_status = "Moderate Perplexity (Standard Academic)"
    else:
        perplexity_status = "Low Perplexity (Predictable / Formulaic)"

    # AI Transition & Fingerprint Overuse Detector
    lower_text = text.lower()
    transition_count = sum(len(re.findall(r'\b' + re.escape(tw) + r'\b', lower_text)) for tw in AI_TRANSITION_WORDS)
    ai_density_per_k = round((transition_count / max(1, len(words))) * 1000.0, 1)

    return {
        "burstiness_score": round(burstiness_cv, 3),
        "burstiness_status": burstiness_status,
        "burstiness_color": burstiness_color,
        "perplexity_entropy": entropy,
        "perplexity_status": perplexity_status,
        "ai_transition_count": transition_count,
        "ai_density_per_k": ai_density_per_k,
        "sentence_len_mean": round(mean_len, 1),
        "sentence_len_std": round(std_len, 1),
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

        # If sentence counts are identical, enforce strict 1-to-1 positional correspondence
        if n1 == n2:
            return list(range(n1))

        # If sentence counts differ, combine positional document progress with semantic similarity
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

        # Compute Burstiness & Perplexity metrics
        bp_orig = compute_burstiness_and_perplexity(orig_text)
        bp_para = compute_burstiness_and_perplexity(para_text)

        # Document Level Semantic Similarity
        doc_matrix = self.compute_similarity_matrix([orig_text], [para_text])
        semantic_sim = float(np.clip(doc_matrix[0, 0], 0.0, 1.0))

        # Lexical Replacement Rate (1 - Jaccard overlap)
        lexical_replacement_rate = 1.0 - jaccard_sim

        # Multi-factor Turnitin AI Detector & Plagiarism Risk Index (0-100%)
        # 1. Direct Lexical Overlap Component
        lexical_risk = jaccard_sim * 100.0
        
        # 2. AI Monotone Cadence Penalty (Low Burstiness CV < 0.45 triggers AI detectors)
        burst_cv = bp_para.get("burstiness_score", 0.4)
        burst_penalty = max(0.0, (0.45 - burst_cv) * 25.0) if burst_cv < 0.45 else 0.0
        
        # 3. AI Fingerprint Vocabulary Overuse Penalty
        ai_density = bp_para.get("ai_density_per_k", 0.0)
        fp_penalty = min(20.0, ai_density * 2.0)

        # Composite Turnitin Risk Score (Combining Plagiarism Overlap + AI Statistical Signatures)
        turnitin_risk_pct = round(min(100.0, lexical_risk + burst_penalty + fp_penalty), 1)
        turnitin_compliant = bool(turnitin_risk_pct <= 10.0)

        # Status & Color Evaluation
        if turnitin_risk_pct > 25.0 or jaccard_sim >= 0.25:
            status_title = f"Turnitin Risk: High ({turnitin_risk_pct}% Composite Risk)"
            status_desc = f"Composite risk is {turnitin_risk_pct}%. Lexical overlap ({int(jaccard_sim*100)}%), low sentence length variance, or AI fingerprint terms will trigger Turnitin. Further re-wording is required."
            status_color = "red"
            status_badge = "🔴 Exceeds Turnitin Cutoff (>10%)"
        elif turnitin_risk_pct > 10.0:
            status_title = f"Turnitin Risk: Moderate ({turnitin_risk_pct}% Composite Risk)"
            status_desc = f"Good conceptual alignment ({int(semantic_sim*100)}%), but composite risk ({turnitin_risk_pct}%) slightly exceeds the <10% submission threshold. Recommended: vary sentence lengths and replace AI transition terms."
            status_color = "orange"
            status_badge = "🟡 Near Turnitin Target (10-25%)"
        else:
            status_title = "Turnitin Compliant (<10% Plagiarism & AI Risk Target Met)"
            status_desc = f"Excellent human-like prose transformation! Composite Turnitin risk is {turnitin_risk_pct}%, safely meeting the <10% submission threshold while preserving core technical intent ({int(semantic_sim*100)}%)."
            status_color = "green"
            status_badge = "🟢 Turnitin Compliant (<10%)"

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

                sentence_analysis.append({
                    "orig_id": i + 1,
                    "orig_sentence": s_orig,
                    "best_match_id": best_idx + 1,
                    "best_match_sentence": best_sent_match,
                    "similarity": round(best_sim, 4),
                    "lexical_overlap": round(s_lexical_sim, 4),
                    "risk_level": risk_level,
                    "risk_badge": risk_badge
                })

        # 4. Compute PCA Vector Embedding Graph Projections
        pca_embedding_data = self.compute_pca_embedding_space(orig_sents, para_sents)

        all_sent_sims = [s["similarity"] for s in sentence_analysis] if sentence_analysis else [0.0]

        # Composite Turnitin AI Detector & Plagiarism Risk Index
        composite_turnitin_risk_pct = round(min(100.0, max(lexical_risk + burst_penalty + fp_penalty, (verbatim_copy_count / max(1, len(orig_sents))) * 100.0)), 1)
        turnitin_compliant = composite_turnitin_risk_pct <= 10.0 and verbatim_copy_count == 0

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
                "turnitin_risk_pct": composite_turnitin_risk_pct,
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
                    "turnitin_risk_pct": turnitin_risk_pct,
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
