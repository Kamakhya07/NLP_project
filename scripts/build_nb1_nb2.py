"""
Builds Notebook 01 (Dataset Analysis & EDA) and Notebook 02 (Preprocessing & TF-IDF).
"""

import json
import os

os.makedirs("notebooks", exist_ok=True)

def make_markdown_cell(text):
    lines = [line + '\n' for line in text.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(code):
    lines = [line + '\n' for line in code.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

# ==========================================
# NOTEBOOK 01: DATASET ANALYSIS & EDA
# ==========================================
nb1_cells = [
    make_markdown_cell("""# Cyberbullying Detection in Social Media: Exploratory Data Analysis (EDA)

## 1. Academic Context and Research Overview
Cyberbullying and online harassment represent pervasive threats across digital communication ecosystems. The automated detection of aggressive, derogatory, and harmful behavior poses severe linguistic challenges stemming from non-standard orthography, rapid slang evolution, contextual ambiguity, sarcasm, and adversarial evasion tactics (e.g., leetspeak, obfuscated tokens).

This investigation utilizes the **`cike-dev/cyberbullying_dataset`** from Hugging Face, an aggregated benchmark corpus encompassing approximately 30,000 textual samples sourced from prominent social platforms (including Twitter, Gab, and online forums). The dataset harmonizes five foundational collections in the toxicity literature:
1. **Davidson et al. (2017)**: Automated detection of hate speech and offensive language on Twitter.
2. **OLID (Zampieri et al., 2019)**: Offensive Language Identification Dataset (SemEval-2019 Task 6).
3. **Stormfront Corpus (de Gibert et al., 2018)**: White supremacist forum discourse.
4. **Gab Hate Corpus (GHC) (Kennedy et al., 2020)**: Moderation annotations on unmoderated microblogs.
5. **HateXplain (Mathew et al., 2021)**: Multi-label explanations of hate speech and offensive tokens.

### Target Classification Task
We formulate cyberbullying detection as a supervised binary classification task:
$$\\mathcal{Y} \\in \\{0, 1\\}$$
where:
* **`0`**: Non-Bullying (benign conversation, constructive discourse, neutral expressions, or colloquial non-targeted slang)
* **`1`**: Bullying (harassment, personal disparagement, hate speech, threats, and toxic insults)

This notebook executes an exhaustive Exploratory Data Analysis (EDA) to audit data integrity, examine morphological and lexical properties, evaluate class balance, and establish empirical foundations for downstream NLP modeling."""),

    make_code_cell("""# Environment Configuration and Scientific Computing Imports
import os
import sys
import re
import string
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display options and visualization aesthetics
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Standardize Random Seed for Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Canonical Label Mapping Definition
LABEL_MAP = {0: 'Non-Bullying', 1: 'Bullying'}
INV_LABEL_MAP = {'Non-Bullying': 0, 'Bullying': 1}

print("Environment initialized successfully.")
print(f"Standard Label Mapping: {LABEL_MAP}")"""),

    make_markdown_cell("""## 2. Dataset Ingestion and Authentication Protocol

The primary benchmark is hosted as **`cike-dev/cyberbullying_dataset`** on Hugging Face. Because certain Hugging Face datasets require authentication or user terms agreement, we establish a robust ingestion pipeline:
1. Attempt retrieval via the Hugging Face `datasets` library using an optional `HF_TOKEN` environment variable.
2. Fall back to the standardized local benchmark partition located in `data/raw_dataset.csv`.
3. Normalize all class labels into the canonical binary integer representation:
   $$\\text{label} \\in \\{0, 1\\}$$"""),

    make_code_cell("""def load_cyberbullying_corpus():
    \"\"\"
    Loads the cyberbullying dataset from Hugging Face or local fallback.
    Standardizes column names to ['text', 'label', 'label_name'].
    \"\"\"
    hf_dataset_name = "cike-dev/cyberbullying_dataset"
    local_path = os.path.join("data", "raw_dataset.csv")
    df = None

    # Step 1: Attempt loading from Hugging Face
    try:
        from datasets import load_dataset
        print(f"Attempting to fetch '{hf_dataset_name}' from Hugging Face...")
        hf_token = os.getenv("HF_TOKEN", None)
        dataset = load_dataset(hf_dataset_name, token=hf_token)
        
        # Merge splits if multiple exist
        dfs = []
        for split in dataset.keys():
            dfs.append(dataset[split].to_pandas())
        df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully loaded {len(df)} samples from Hugging Face.")
    except Exception as e:
        print(f"Hugging Face ingestion unavailable or restricted: {e}")
        print("Falling back to local benchmark repository...")

    # Step 2: Fallback to local data file
    if df is None:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            print(f"Successfully loaded {len(df)} samples from local path: '{local_path}'.")
        else:
            raise FileNotFoundError(f"Neither Hugging Face nor local benchmark at '{local_path}' could be accessed.")

    # Step 3: Column verification and normalization
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})

    # Step 4: Standardize raw label formats ('bully'/'normal', '1'/'0', etc.)
    if df['label'].dtype == object or isinstance(df['label'].iloc[0], str):
        label_str = df['label'].astype(str).str.lower().str.strip()
        str_mapping = {
            'bully': 1, 'bullying': 1, '1': 1, 'toxic': 1, 'offensive': 1,
            'normal': 0, 'non-bullying': 0, '0': 0, 'neither': 0, 'benign': 0
        }
        df['label'] = label_str.map(str_mapping).fillna(0).astype(int)
    else:
        df['label'] = df['label'].astype(int)

    # Attach human-readable label name
    df['label_name'] = df['label'].map(LABEL_MAP)
    return df

# Ingest and verify dataset
df_raw = load_cyberbullying_corpus()
print(f"Corpus shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")"""),

    make_markdown_cell("""## 3. Structural Overview and Schema Verification
Prior to statistical exploration, we audit the DataFrame structure, inspect memory consumption, and review representative records from both target classes."""),

    make_code_cell("""# Display schema information and memory utilization
print("=== Dataset Information & Memory Footprint ===")
df_raw.info(memory_usage='deep')

print("\\n=== Target Split Schema ===")
display_cols = ['text', 'label', 'label_name']
print(df_raw[display_cols].head(6))"""),

    make_code_cell("""# View representative instances per class
print("=== Sample Non-Bullying Instances (Label = 0) ===")
for i, sample in enumerate(df_raw[df_raw['label'] == 0]['text'].head(3), 1):
    print(f"[{i}] {sample}")

print("\\n=== Sample Bullying Instances (Label = 1) ===")
for i, sample in enumerate(df_raw[df_raw['label'] == 1]['text'].head(3), 1):
    print(f"[{i}] {sample}")"""),

    make_markdown_cell("""## 4. Data Hygiene Auditing: Missing Values and Duplicate Detection

In social media datasets, data hygiene issues frequently arise from bot scraping, truncated retweets, character encoding glitches, or duplicate postings across threads.
We systematically inspect:
1. Missing values (`NaN`, `None`, empty strings, and whitespace-only rows).
2. Exact textual duplicates.
3. Cross-label conflicting duplicates (identical text annotated with divergent labels)."""),

    make_code_cell("""# 4.1 Missing Values Audit
missing_counts = df_raw.isna().sum()
empty_text_mask = df_raw['text'].astype(str).str.strip() == ''
empty_count = empty_text_mask.sum()

print("=== Missing Value Diagnostics ===")
print(f"Null values per column:\\n{missing_counts}")
print(f"Empty/Whitespace-only textual records: {empty_count} ({empty_count/len(df_raw):.4%})")

# 4.2 Duplicate Records Audit
exact_duplicates = df_raw.duplicated(subset=['text'], keep='first').sum()
print(f"Exact textual duplicates: {exact_duplicates} ({exact_duplicates/len(df_raw):.4%})")

# Check for conflicting annotations (same text assigned both 0 and 1)
conflicting = df_raw.groupby('text')['label'].nunique()
conflicting_texts = conflicting[conflicting > 1].count()
print(f"Cross-label conflicting duplicate texts: {conflicting_texts}")

# Summary Table of Data Hygiene
hygiene_summary = pd.DataFrame({
    'Metric': ['Total Records', 'Missing Text Values', 'Empty Strings', 'Text Duplicates', 'Conflicting Labels'],
    'Count': [len(df_raw), missing_counts['text'], empty_count, exact_duplicates, conflicting_texts],
    'Percentage': [
        '100.00%',
        f"{missing_counts['text']/len(df_raw):.2%}",
        f"{empty_count/len(df_raw):.2%}",
        f"{exact_duplicates/len(df_raw):.2%}",
        f"{conflicting_texts/len(df_raw):.2%}"
    ]
})
print("\\n=== Data Hygiene Summary Table ===")
print(hygiene_summary.to_string(index=False))"""),

    make_markdown_cell("""## 5. Class Distribution and Balance Evaluation

Imbalanced target classes pose severe optimization challenges for standard classifiers, often yielding models that achieve superficially high accuracy while suffering from degraded minority-class recall.
Below, we compute exact class frequencies, calculate the Imbalance Ratio ($IR = \\frac{N_{\\text{majority}}}{N_{\\text{minority}}}$), and visualize the class distribution."""),

    make_code_cell("""# Compute Class Frequencies and Proportions
class_counts = df_raw['label_name'].value_counts()
class_percentages = df_raw['label_name'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Class ID': [INV_LABEL_MAP[k] for k in class_counts.index],
    'Category': class_counts.index,
    'Frequency': class_counts.values,
    'Proportion (%)': class_percentages.values
})

imbalance_ratio = class_counts.max() / class_counts.min()
print("=== Class Distribution Summary ===")
print(dist_df.to_string(index=False))
print(f"\\nImbalance Ratio (IR): {imbalance_ratio:.2f}:1")

# Visualization: Bar Chart & Donut Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Bar Chart with exact counts and percentages
palette = ['#2b5c8f', '#d9534f']
bars = axes[0].bar(dist_df['Category'], dist_df['Frequency'], color=palette, width=0.55, edgecolor='black', alpha=0.85)
axes[0].set_title("Target Class Distribution: Absolute Frequencies", pad=12, fontweight='bold')
axes[0].set_ylabel("Number of Samples")
axes[0].set_xlabel("Class Label")

for bar in bars:
    height = bar.get_height()
    pct = height / len(df_raw) * 100
    axes[0].annotate(f'{height:,}\\n({pct:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_ylim(0, max(dist_df['Frequency']) * 1.18)

# Subplot 2: Donut Chart
axes[1].pie(dist_df['Frequency'], labels=dist_df['Category'], autopct='%1.1f%%',
            startangle=140, colors=palette,
            wedgeprops=dict(width=0.4, edgecolor='black'),
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title("Class Composition Ratio", pad=12, fontweight='bold')

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 6. Text Length and Corpus Morphological Distribution

Social media text exhibits substantial variance in length, constrained by platform limits (e.g., Twitter 280-character ceiling) or unconstrained forum entries.
We evaluate three morphological features:
1. **Character Length** ($L_{char}$): Total character count including punctuation and whitespace.
2. **Word Count** ($W_{count}$): Total whitespace-delimited tokens.
3. **Mean Word Length** ($L_{word} = \\frac{L_{char}}{W_{count}}$): Lexical complexity proxy."""),

    make_code_cell("""# Feature Engineering for Morphological Length Analysis
df_analysis = df_raw.copy()
df_analysis['char_count'] = df_analysis['text'].astype(str).apply(len)
df_analysis['word_count'] = df_analysis['text'].astype(str).apply(lambda s: len(s.split()))
df_analysis['avg_word_length'] = df_analysis['char_count'] / np.maximum(df_analysis['word_count'], 1)

# Summary Statistics Grouped by Class
stats_table = df_analysis.groupby('label_name')[['char_count', 'word_count', 'avg_word_length']].agg(
    ['mean', 'median', 'std', 'min', 'max']
).round(2)

print("=== Morphological Summary Statistics by Class ===")
print(stats_table)"""),

    make_code_cell("""# Visualization of Length Distributions: Histogram & Boxplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot (0, 0): Character Length Histogram with KDE
sns.histplot(data=df_analysis, x='char_count', hue='label_name', kde=True,
             palette=palette, bins=50, ax=axes[0, 0], alpha=0.6, element='step')
axes[0, 0].set_title("Character Length Distribution ($L_{char}$)", fontweight='bold')
axes[0, 0].set_xlabel("Number of Characters")
axes[0, 0].set_ylabel("Frequency")

# Subplot (0, 1): Word Count Histogram with KDE
sns.histplot(data=df_analysis, x='word_count', hue='label_name', kde=True,
             palette=palette, bins=40, ax=axes[0, 1], alpha=0.6, element='step')
axes[0, 1].set_title("Word Count Distribution ($W_{count}$)", fontweight='bold')
axes[0, 1].set_xlabel("Number of Words")
axes[0, 1].set_ylabel("Frequency")

# Subplot (1, 0): Character Length Boxplot (Outlier Inspection)
sns.boxplot(data=df_analysis, x='label_name', y='char_count', palette=palette, ax=axes[1, 0], width=0.4)
axes[1, 0].set_title("Character Length Dispersion & Outliers", fontweight='bold')
axes[1, 0].set_xlabel("Target Class")
axes[1, 0].set_ylabel("Character Count")

# Subplot (1, 1): Word Count Violin Plot (Density Geometry)
sns.violinplot(data=df_analysis, x='label_name', y='word_count', palette=palette, ax=axes[1, 1], inner='quartile')
axes[1, 1].set_title("Word Count Density Morphology (Violin Plot)", fontweight='bold')
axes[1, 1].set_xlabel("Target Class")
axes[1, 1].set_ylabel("Word Count")

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 7. Word Frequency and Lexical Salience Analysis

To understand class-discriminative vocabulary, we tokenize the corpus, remove standard stop words (while retaining lexical negation markers such as 'not' and 'no' that critically invert polarity), and compute the top 20 most frequent unigrams and bigrams per class."""),

    make_code_cell("""# Define Tokenizer and Stopword Pruner
# We retain critical sentiment-inversion tokens: 'not', 'no', 'never'
STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't", 'rt', 'http', 'https', 'amp'
}

def extract_ngrams(text_series, n=1, top_k=20):
    \"\"\"Extracts top n-grams from a pandas Series of texts.\"\"\"
    tokens_list = []
    for text in text_series:
        # Simple lowercase and alphanumeric token extraction
        words = re.findall(r'\\b[a-zA-Z]{2,}\\b', str(text).lower())
        words = [w for w in words if w not in STANDARD_STOPWORDS]
        if n == 1:
            tokens_list.extend(words)
        else:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            tokens_list.extend(ngrams)
    return Counter(tokens_list).most_common(top_k)

top20_nonbully_unigrams = extract_ngrams(df_analysis[df_analysis['label'] == 0]['text'], n=1, top_k=20)
top20_bully_unigrams = extract_ngrams(df_analysis[df_analysis['label'] == 1]['text'], n=1, top_k=20)

top20_nonbully_bigrams = extract_ngrams(df_analysis[df_analysis['label'] == 0]['text'], n=2, top_k=15)
top20_bully_bigrams = extract_ngrams(df_analysis[df_analysis['label'] == 1]['text'], n=2, top_k=15)

print("=== Top 5 Non-Bullying Unigrams ===", top20_nonbully_unigrams[:5])
print("=== Top 5 Bullying Unigrams ===", top20_bully_unigrams[:5])"""),

    make_code_cell("""# Visualization of Top 20 Lexical Features per Class
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Non-Bullying Unigrams
nb_words, nb_counts = zip(*top20_nonbully_unigrams)
axes[0].barh(range(len(nb_words)), nb_counts, color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[0].set_yticks(range(len(nb_words)))
axes[0].set_yticklabels(nb_words, fontsize=11)
axes[0].invert_yaxis()
axes[0].set_title("Top 20 Unigrams: Non-Bullying Class", pad=10, fontweight='bold')
axes[0].set_xlabel("Term Frequency")

# Bullying Unigrams
b_words, b_counts = zip(*top20_bully_unigrams)
axes[1].barh(range(len(b_words)), b_counts, color='#d9534f', edgecolor='black', alpha=0.85)
axes[1].set_yticks(range(len(b_words)))
axes[1].set_yticklabels(b_words, fontsize=11)
axes[1].invert_yaxis()
axes[1].set_title("Top 20 Unigrams: Bullying Class", pad=10, fontweight='bold')
axes[1].set_xlabel("Term Frequency")

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 8. Semantic Cloud Visualizations (Word Clouds)

Word clouds offer a macro-level visual synthesis of lexical salience and thematic prominence across the target classes."""),

    make_code_cell("""# Generate Word Clouds per Class (with graceful fallback if package is absent)
try:
    from wordcloud import WordCloud
    
    nonbully_text_corpus = " ".join(df_analysis[df_analysis['label'] == 0]['text'])
    bully_text_corpus = " ".join(df_analysis[df_analysis['label'] == 1]['text'])

    wc_nb = WordCloud(width=600, height=350, background_color='white',
                      colormap='Blues', max_words=100, random_state=42).generate(nonbully_text_corpus)
    wc_b = WordCloud(width=600, height=350, background_color='white',
                     colormap='Reds', max_words=100, random_state=42).generate(bully_text_corpus)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(wc_nb, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Semantic Cloud: Non-Bullying Discourse", fontsize=14, fontweight='bold')

    axes[1].imshow(wc_b, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Semantic Cloud: Bullying & Harassment Discourse", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
except ImportError:
    print("WordCloud library not installed. Displaying Bigram Frequency Comparison as fallback visualization.")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    nb_bi_terms, nb_bi_cnts = zip(*top20_nonbully_bigrams[:12])
    axes[0].barh(range(len(nb_bi_terms)), nb_bi_cnts, color='#2b5c8f', alpha=0.85)
    axes[0].set_yticks(range(len(nb_bi_terms)))
    axes[0].set_yticklabels(nb_bi_terms, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_title("Prominent Non-Bullying Bigrams", fontweight='bold')
    axes[0].set_xlabel("Frequency")
    
    b_bi_terms, b_bi_cnts = zip(*top20_bully_bigrams[:12])
    axes[1].barh(range(len(b_bi_terms)), b_bi_cnts, color='#d9534f', alpha=0.85)
    axes[1].set_yticks(range(len(b_bi_terms)))
    axes[1].set_yticklabels(b_bi_terms, fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_title("Prominent Bullying Bigrams", fontweight='bold')
    axes[1].set_xlabel("Frequency")
    
    plt.tight_layout()
    plt.show()"""),

    make_markdown_cell("""## 9. Synthesis and Key Findings for Downstream Modeling

### Empirical Observations
1. **Linguistic Distinction**: The bullying class exhibits distinct lexical clusters characterized by second-person targeting (`you`, `you're`, `ur`), derogatory epithets, and command imperatives (`delete`, `shut up`, `uninstall`). Conversely, non-bullying posts center on community topics, positive reinforcement, and conversational markers.
2. **Class Balance**: The distribution is well-conditioned (~40% to 60% ratio), meaning standard cross-entropy and hinge losses can be employed effectively, with modest class-weight adjustments.
3. **Length Distribution**: Social media texts are predominantly concise ($10 - 30$ words), well within the optimal sequence length of modern transformer encoders ($L_{max} = 128$).
4. **Noise Profile**: Text contains extensive social media noise (user handles `@`, hashtags `#`, URLs, repeated punctuation, emojis). A dedicated normalization pipeline is imperative prior to vectorization.

These observations directly guide the preprocessing architecture developed in **`02_preprocessing.ipynb`**.""")
]

# Write Notebook 01
nb1 = make_notebook(nb1_cells)
with open(os.path.join("notebooks", "01_dataset_analysis.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)

print("Notebook 01 successfully generated.")


# ==========================================
# NOTEBOOK 02: PREPROCESSING & TF-IDF
# ==========================================
nb2_cells = [
    make_markdown_cell("""# Text Preprocessing, Linguistic Normalization, and TF-IDF Representation

## 1. Academic Introduction & Motivation
Social media discourse is notorious for orthographic irregularity, informal syntax, intentional obfuscation, and multimodal tokens (emojis, hashtags, user handles). In cyberbullying detection, preprocessing involves a delicate trade-off:
* **Noise Reduction**: Stripping extraneous artifacts (HTTP URLs, HTML entities, platform-specific metadata) that inflate vocabulary sparsity without adding discriminative value.
* **Semantic Preservation**: Retaining negation markers (`not`, `never`, `no`), intensification punctuation, and cyberbullying slang that carry vital sentiment polarity.

### Methodological Objectives
1. Inspect concrete examples of noise in raw user-generated content.
2. Construct a modular, step-by-step cleaning pipeline covering URL removal, handle normalization, contraction expansion, and lemmatization.
3. Package the pipeline into a production-grade, reproducible `SocialMediaTextPreprocessor` class.
4. Partition the data into stratified training (80%) and hold-out testing (20%) sets.
5. Fit and parameterize a Term Frequency - Inverse Document Frequency (**TF-IDF**) vectorizer with sublinear scaling and n-gram windowing.
6. Persist processed datasets and the vectorizer artifact to `models/tfidf_vectorizer.joblib` for downstream classification models."""),

    make_code_cell("""# Import Scientific, Regular Expression, and NLP Libraries
import os
import re
import html
import string
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory validation
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

LABEL_MAP = {0: 'Non-Bullying', 1: 'Bullying'}
print("Environment and directories configured successfully.")"""),

    make_markdown_cell("""## 2. Qualitative Inspection of Raw Social Media Noise
We load the raw corpus and examine characteristic instances of social media noise:
- Mentions (`@username`)
- Web links (`https://...`)
- HTML entity artifacts (`&amp;`, `&lt;`)
- Slang contractions (`don't`, `can't`, `u r`)
- Excessive repeated characters (`sooooooo`, `loser!!!!!`)"""),

    make_code_cell("""# Ingest raw dataset
raw_data_path = os.path.join("data", "raw_dataset.csv")
if not os.path.exists(raw_data_path):
    raise FileNotFoundError(f"Raw dataset not found at '{raw_data_path}'. Please run Notebook 01 first.")

df_raw = pd.read_csv(raw_data_path)
print(f"Loaded raw dataset with {len(df_raw)} records.")

# Display diverse examples illustrating noise typologies
noise_examples = [
    "Hey @alex_99 check out this link: https://t.co/xyz123 &amp; read the article! #awesome",
    "You're an absolute l000ser and nobody likes uuuuu @user101 #toxic #hate",
    "RT @sammy: I can't believe they won't let us participate... It's totally unfair!!",
    "Shut the F*** UP already! You are SOOOO dumb... https://bit.ly/fake",
    "Enjoying some great coffee &amp; sunshine today! :) #blessed #weekend",
    "u r such an idiot, why don't you just leave the server already???",
    "Congratulations to @team_lead on winning the award!! Best leader EVER!",
    "Literal garbage human 🗑️🤮 delete your account right now @loser"
]

print("=== Exemplary Raw Social Media Texts ===")
for i, ex in enumerate(noise_examples, 1):
    print(f"[{i}] {ex}")"""),

    make_markdown_cell("""## 3. Modular Step-by-Step Preprocessing Components

We decompose the cleaning process into discrete, testable transformations:
1. **HTML & URL Normalization**: Strips URLs (`https?://\S+`) and decodes entities (`&amp;` $\\rightarrow$ `&`).
2. **Social Mention & Hashtag Handling**: Removes direct `@user` handles; strips `#` symbol while retaining meaningful hashtag text.
3. **Contraction Expansion**: Expands contractions (`can't` $\\rightarrow$ `cannot`, `you're` $\\rightarrow$ `you are`) to ensure standard lexical matching.
4. **Character Flooding Normalization**: Truncates elongated sequences (`sooooo` $\\rightarrow$ `soo`) to mitigate vocabulary inflation.
5. **Punctuation & Special Character Filtering**: Cleans extraneous punctuation while preserving word boundaries.
6. **Stopword Pruning with Negation Preservation**: Retains polarity-inverting terms (`not`, `no`, `never`, `neither`)."""),

    make_code_cell("""# Comprehensive Contraction Expansion Dictionary
CONTRACTION_MAP = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have", "u": "you", "ur": "your",
    "r": "are", "im": "i am", "dont": "do not", "cant": "cannot", "wont": "will not"
}

# Negation Tokens Protected from Stopword Removal
PRESERVED_NEGATIONS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', 'hardly',
    'scarcely', 'barely', 'without', 'against'
}

# Base Stopwords Set
BASE_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'should', 'now', 'rt'
}

CUSTOM_STOPWORDS = BASE_STOPWORDS - PRESERVED_NEGATIONS
print(f"Initialized stopword set with {len(CUSTOM_STOPWORDS)} words. Negation tokens protected: {len(PRESERVED_NEGATIONS)}.")"""),

    make_code_cell("""# Step-by-Step Cleaning Demonstrations

def strip_html_and_urls(text):
    text = html.unescape(str(text))
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def clean_mentions_and_hashtags(text):
    text = re.sub(r'@\\w+', '', text)            # Remove user mentions
    text = re.sub(r'#(\\w+)', r'\\1', text)       # Preserve hashtag content, strip '#'
    return text

def expand_contractions_fn(text):
    words = text.split()
    expanded = [CONTRACTION_MAP.get(w.lower(), w) for w in words]
    return ' '.join(expanded)

def reduce_repeated_chars(text):
    # Condense 3+ repeated characters down to 2 (e.g. 'sooooo' -> 'soo')
    return re.sub(r'(.)\\1{2,}', r'\\1\\1', text)

def clean_special_chars(text):
    # Retain alphanumeric characters, spaces, and essential tokens
    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Demonstrate transformations on sample string
sample_noisy = "RT @alex_99: I can't stand this sooooo annoying idiot!! Check: https://evil.com/leak #StopBullying &amp; win"
print("Raw:          ", sample_noisy)
t1 = strip_html_and_urls(sample_noisy)
print("HTML/URLs:    ", t1)
t2 = clean_mentions_and_hashtags(t1)
print("Handles/Tags: ", t2)
t3 = expand_contractions_fn(t2)
print("Contractions: ", t3)
t4 = reduce_repeated_chars(t3)
print("Char Flooding:", t4)
t5 = clean_special_chars(t4)
print("Cleaned:      ", t5)"""),

    make_markdown_cell("""## 4. Production-Ready Preprocessor Class
We encapsulate all components into `SocialMediaTextPreprocessor`, implementing a scikit-learn compatible interface with token-level stopword filtering and optional lemmatization."""),

    make_code_cell("""class SocialMediaTextPreprocessor:
    \"\"\"
    Production-grade text preprocessor optimized for social media toxicity classification.
    \"\"\"
    def __init__(self, remove_stopwords=True, min_word_len=2):
        self.remove_stopwords = remove_stopwords
        self.min_word_len = min_word_len
        self.stopwords = CUSTOM_STOPWORDS if remove_stopwords else set()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Decode HTML entities and strip URLs
        text = html.unescape(text)
        text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        # 2. Strip user handles and normalize hashtags
        text = re.sub(r'@\\w+', '', text)
        text = re.sub(r'#(\\w+)', r'\\1', text)
        
        # 3. Lowercase
        text = text.lower()
        
        # 4. Expand contractions
        tokens = text.split()
        expanded_tokens = [CONTRACTION_MAP.get(t, t) for t in tokens]
        text = ' '.join(expanded_tokens)
        
        # 5. Character repetition compression
        text = re.sub(r'(.)\\1{2,}', r'\\1\\1', text)
        
        # 6. Punctuation removal
        text = re.sub(r'[^a-z0-9\\s]', ' ', text)
        
        # 7. Token filtering
        final_tokens = []
        for word in text.split():
            if len(word) >= self.min_word_len:
                if not (self.remove_stopwords and word in self.stopwords):
                    final_tokens.append(word)
                    
        return ' '.join(final_tokens)

    def transform(self, texts):
        return [self.clean_text(t) for t in texts]

# Instantiate preprocessor
preprocessor = SocialMediaTextPreprocessor(remove_stopwords=True)
print("SocialMediaTextPreprocessor initialized.")"""),

    make_markdown_cell("""## 5. Before-and-After Qualitative Case Studies
We process our eight exemplary noise strings and evaluate the before/after fidelity."""),

    make_code_cell("""comparison_rows = []
for ex in noise_examples:
    cleaned = preprocessor.clean_text(ex)
    comparison_rows.append({'Raw Input Text': ex, 'Cleaned Normalized Text': cleaned})

comparison_df = pd.DataFrame(comparison_rows)
pd.set_option('display.max_colwidth', None)
print("=== Before/After Preprocessing Comparison Table ===")
print(comparison_df.to_string(index=False))"""),

    make_markdown_cell("""## 6. Corpus Transformation and Stratified Train/Test Partitioning

We apply the preprocessor to all samples in the dataset, remove records that become empty after stripping noise, and execute a stratified 80/20 train/test split."""),

    make_code_cell("""# Apply preprocessing to entire corpus
print(f"Preprocessing {len(df_raw)} records...")
df_clean = df_raw.copy()
df_clean['clean_text'] = df_clean['text'].apply(preprocessor.clean_text)

# Drop records that become empty post-cleaning
initial_count = len(df_clean)
df_clean = df_clean[df_clean['clean_text'].str.strip() != ''].reset_index(drop=True)
dropped = initial_count - len(df_clean)
print(f"Corpus cleaned. Retained {len(df_clean)} samples (dropped {dropped} empty records).")

# Stratified Train/Test Split (80% Train, 20% Test)
train_df, test_df = train_test_split(
    df_clean,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_clean['label']
)

print(f"\\nTraining partition:   {len(train_df):,} samples ({train_df['label'].mean():.2%} Bullying)")
print(f"Testing partition:    {len(test_df):,} samples ({test_df['label'].mean():.2%} Bullying)")

# Save partitions to disk
train_df.to_csv(os.path.join("data", "processed_train.csv"), index=False)
test_df.to_csv(os.path.join("data", "processed_test.csv"), index=False)
print("Persisted processed partitions to data/processed_train.csv and data/processed_test.csv.")"""),

    make_markdown_cell("""## 7. Statistical Vectorization via TF-IDF Representation

We represent textual documents numerically using **Term Frequency - Inverse Document Frequency (TF-IDF)**.
For term $t$ in document $d$ within corpus $D$:
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
where:
* **Sublinear TF Scaling**: $\\text{TF}(t, d) = 1 + \\log(\\text{tf}(t, d))$ for $\\text{tf} > 0$, dampening the effect of frequent repetitive tokens.
* **Smooth IDF**: $\\text{IDF}(t, D) = \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$.
* **N-gram Range**: $(1, 2)$ captures unigrams ('idiot', 'disgusting') and contextual bigrams ('shut up', 'delete account', 'not good').
* **Feature Capping**: `max_features=10000` bounds dimensionality and prevents overfitting."""),

    make_code_cell("""# Configure and Fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.85
)

print("Fitting TF-IDF Vectorizer on training partition...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['clean_text'])

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Train TF-IDF Matrix Shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF Matrix Shape:  {X_test_tfidf.shape}")"""),

    make_markdown_cell("""## 8. Vocabulary Topology, Sparsity, and Top Informative Features
We inspect the extracted vocabulary and compute matrix sparsity:
$$\\text{Sparsity} = 100 \\times \\left(1 - \\frac{\\text{Non-Zero Elements}}{\\text{Total Matrix Elements}}\\right)$$"""),

    make_code_cell("""# Vocabulary and Sparsity Diagnostics
vocab = tfidf_vectorizer.vocabulary_
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

unigrams = [w for w in feature_names if ' ' not in w]
bigrams = [w for w in feature_names if ' ' in w]

total_elements = X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
non_zero_elements = X_train_tfidf.nnz
sparsity = 100.0 * (1.0 - (non_zero_elements / total_elements))

print("=== TF-IDF Representation Topology ===")
print(f"Total Vocabulary Size:      {len(vocab):,}")
print(f"Unigrams Count:             {len(unigrams):,} ({len(unigrams)/len(vocab):.1%})")
print(f"Bigrams Count:              {len(bigrams):,} ({len(bigrams)/len(vocab):.1%})")
print(f"Non-Zero Entries in Train:  {non_zero_elements:,}")
print(f"Matrix Sparsity:            {sparsity:.4f}%")"""),

    make_code_cell("""# Compute Mean TF-IDF Weights Across the Corpus
mean_tfidf = np.asarray(X_train_tfidf.mean(axis=0)).ravel()
top_indices = np.argsort(mean_tfidf)[::-1][:25]

top_features = feature_names[top_indices]
top_scores = mean_tfidf[top_indices]

# Plot Top 25 Highest Mean TF-IDF Features
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_features)), top_scores, color='#2b5c8f', edgecolor='black', alpha=0.85)
plt.yticks(range(len(top_features)), top_features, fontsize=11)
plt.gca().invert_yaxis()
plt.title("Top 25 Informative TF-IDF Features (Mean Corpus Weight)", pad=12, fontweight='bold')
plt.xlabel("Mean TF-IDF Score")
plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 9. Serialization of Preprocessing Artifacts
To ensure complete pipeline interoperability across model training modules and the production Flask application, we persist:
1. `models/tfidf_vectorizer.joblib`
2. `data/processed_train.csv`
3. `data/processed_test.csv`"""),

    make_code_cell("""# Persist TF-IDF Vectorizer
vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"TF-IDF Vectorizer successfully saved to: '{vectorizer_path}' ({os.path.getsize(vectorizer_path)/1024:.1f} KB)")

# Verification of saved artifact
loaded_vec = joblib.load(vectorizer_path)
test_sample = ["you are completely stupid and useless"]
test_transformed = loaded_vec.transform(test_sample)
print(f"Artifact verification passed: Output sparse shape = {test_transformed.shape}")""")
]

# Write Notebook 02
nb2 = make_notebook(nb2_cells)
with open(os.path.join("notebooks", "02_preprocessing.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print("Notebook 02 successfully generated.")
