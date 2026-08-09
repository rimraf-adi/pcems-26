# Multi-Dialect Marathi Text Normalization via LLM-Driven Synthetic Augmentation and Neural Sequence-to-Sequence Benchmarking

[![LaTeX Paper](https://img.shields.io/badge/Paper-LaTeX-blue.svg)](paper/norm.pdf)
[![Code Base](https://img.shields.io/badge/Repo-dialect--normalisation-green.svg)](https://github.com/rimraf-adi/dialect-normalisation)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

Official LaTeX paper repository for **"Multi-Dialect Marathi Text Normalization via LLM-Driven Synthetic Augmentation and Neural Sequence-to-Sequence Benchmarking"**.

---

## 📌 Abstract

Low-resource regional dialects pose severe challenges for Natural Language Processing (NLP) systems due to significant morpho-syntactic variation, non-standard orthography, and extreme scarcity of annotated parallel corpora. In this paper, we address the task of dialect normalization---converting non-standard regional dialectal text into Standard Written Marathi---focusing on three major Indo-Aryan regional dialects of Maharashtra: **Malvani**, **Ahirani**, and **Varhadi**. 

To overcome severe data scarcity, we introduce an automated synthetic parallel data expansion pipeline combining large language model (LLM) generation using Gemma-2 (9B/27B) with a multi-tier verification engine (rule-based script filtering and LLM semantic auditing), effectively doubling the clean parallel training corpus into verified synthetically expanded pairs across agriculture and finance domains. We conduct a rigorous empirical evaluation of state-of-the-art multilingual sequence-to-sequence transformer architectures---`google/mT5-small` (300M) and `ai4bharat/IndicBART` (244M)---under an 85/15 stratified test split with 5-fold cross-validation across 8 dataset configurations. 

Empirical results demonstrate that `mT5-small` achieves state-of-the-art performance with **69.51 BLEU** and **84.58 chrF++** on the synthetically expanded multi-dialect dataset (+12.39 BLEU over IndicBART), with individual dialect performance reaching **80.99 BLEU** and **91.21 chrF++** on Varhadi. Furthermore, an ablation study on the verification engine proves that training on unverified raw synthetic data causes a catastrophic drop of **-33.41 BLEU points**, highlighting the critical role of multi-tier auditing.

---

## 📂 Repository Structure

```
pcems26/
├── paper/                    # Structured LaTeX paper directory
│   ├── content.tex           # Primary unified paper manuscript (shared content)
│   ├── norm.tex              # Driver template 1: Custom Single-Column Arial format
│   ├── springer.tex          # Driver template 2: Springer LNCS paper format
│   ├── references.bib        # Complete BibTeX citation database
│   ├── norm.pdf              # Pre-compiled PDF (Single-Column Arial Driver)
│   └── springer.pdf          # Pre-compiled PDF (Springer LNCS Driver)
├── build.sh                  # Compilation script (compiles paper/norm.pdf and paper/springer.pdf)
├── dialect-normalisation/    # Dialect normalisation code repository
├── .gitignore                # Git ignore file for LaTeX build artifacts
└── README.md                 # Main repository documentation
```

---

## 🚀 Building the Paper

### Prerequisites

Ensure you have a complete TeX Live / MacTeX distribution with `XeLaTeX` and `BibTeX` installed, along with the required font dependencies:
- `XeLaTeX` (for native Devanagari script and UTF-8 font mapping)
- `BibTeX`
- Arial font (`fontspec` support)
- Devanagari font (`Devanagari Sangam MN` or standard system Devanagari fonts)

### One-Command Build

To automatically compile both `paper/norm.pdf` and `paper/springer.pdf`, execute from the root directory:

```bash
chmod +x build.sh
./build.sh
```

### Manual Compilation

To compile a specific template manually inside the `paper/` directory:

```bash
cd paper

# For Single-Column Arial Template (norm.pdf)
xelatex -interaction=nonstopmode norm.tex
bibtex norm
xelatex -interaction=nonstopmode norm.tex
xelatex -interaction=nonstopmode norm.tex

# For Springer LNCS Template (springer.pdf)
xelatex -interaction=nonstopmode springer.tex
bibtex springer
xelatex -interaction=nonstopmode springer.tex
xelatex -interaction=nonstopmode springer.tex
```

---

## 📊 Key Results Summary

| Model Configuration | Dataset Partition | BLEU | chrF++ | WER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **mT5-small (300M)** | Varhadi (Original) | **80.99** | **91.21** | **14.73** |
| **mT5-small (300M)** | Ahirani (Synthetically Expanded) | **79.76** | **91.50** | **12.61** |
| **IndicBART (244M)** | Multi-Dialect (Synthetically Expanded) | **76.50** | **75.49** | **16.23** |
| **mT5-small (300M)** | **Multi-Dialect (Synthetically Expanded SOTA)** | **73.48** | **87.70** | **16.58** |

---

## 🤝 Code & Data Availability

The underlying codebase for training, evaluation scripts, dataset generation, and pre-trained model weights are hosted in the primary code repository:
👉 **[rimraf-adi/dialect-normalisation](https://github.com/rimraf-adi/dialect-normalisation)**

---

## 📜 Citation

If you use this benchmark, synthetic pipeline, or paper manuscript in your research, please cite:

```bibtex
@article{kinjawadekar2026multidialect,
  title={Multi-Dialect Marathi Text Normalization via LLM-Driven Synthetic Augmentation and Neural Sequence-to-Sequence Benchmarking},
  author={Kinjawadekar, Aditya and Kannaiyan, Surender and Sinha, Saugata},
  journal={Visvesvaraya National Institute of Technology},
  year={2026}
}
```
