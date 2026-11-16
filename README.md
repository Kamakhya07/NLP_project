# 🛡️ CyberShield: Cyberbullying Detection from Social Media

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36%2B-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3.svg?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

**CyberShield** is an end-to-end Natural Language Processing (NLP) system engineered to detect, classify, and mitigate cyberbullying and abusive language across social media interactions. Proliferation of toxic online interactions presents severe psychological and social challenges; automated real-time moderation is critical for protecting online communities and ensuring digital well-being.

This project implements a rigorous comparative study between classical statistical machine learning baselines (**TF-IDF + Logistic Regression**, **TF-IDF + Linear Support Vector Machines**) and state-of-the-art transformer-based deep learning (**Fine-tuned DistilBERT**). The system is deployed via an interactive **Flask** web application backed by an **SQLite** audit database, enabling real-time text analysis, confidence score inspection, historical tracking, and model switching.

---

## 🔬 Research Question

> **"To what extent do lightweight contextual transformer embeddings (DistilBERT) outperform n-gram-based TF-IDF representations combined with linear classifiers (Logistic Regression, Support Vector Machines) in classifying nuanced social media cyberbullying, and how do their computational trade-offs translate to real-time moderation deployment?"**

---

## ✨ Features

- **Multi-Model Architecture**: Seamless inference support across TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and fine-tuned DistilBERT.
- **Robust Text Preprocessing Pipeline**: Handles social media noise, including URL removal, user mention masking (`@user`), hashtag parsing, emoji translation, contraction expansion, and whitespace normalization.
- **Interactive Web Interface**: Clean, accessible UI built with Flask and Bootstrap 5 featuring dynamic confidence gauges, highlight indicators, and instant feedback.
- **Persistent Prediction Logging**: SQLite audit database logs input texts, predicted labels, confidence percentages, response latencies, and active model versions for compliance and analytics.
- **Modular & Production-Ready Codebase**: Structured package architecture (`src/`), centralized configuration (`config.py`), and reproducible data workflows.
- **Comprehensive EDA & Diagnostics**: Dedicated Jupyter notebooks for exploratory data analysis, class balance visualization, n-gram distributions, ROC curves, and confusion matrices.

---

## 🖼️ Screenshots

> *Placeholder: Add application screenshots showcasing the detection interface, model comparisons, and audit log history.*

| Real-Time Detection Dashboard | Audit Log & Analytics |
|:---:|:---:|
| ![Detection Dashboard](reports/figures/screenshot_dashboard_placeholder.png) | ![Audit Log](reports/figures/screenshot_history_placeholder.png) |

---

## 🛠️ Tech Stack

- **Programming Language**: Python 3.10+
- **Deep Learning & Transformers**: [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/transformers/), [Datasets](https://huggingface.co/docs/datasets/)
- **Machine Learning & NLP**: [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [WordCloud](https://github.com/amueller/word_cloud)
- **Web Backend**: [Flask](https://flask.palletsprojects.com/)
- **Frontend**: HTML5, Jinja2, [Bootstrap 5](https://getbootstrap.com/), FontAwesome
- **Database**: SQLite3
- **Interactive Development**: JupyterLab / Notebook

---

## 📂 Project Structure

```text
NLP-project/
│
├── dataset/
│   ├── raw/                           # Raw downloaded datasets
│   │   └── .gitkeep
│   └── processed/                     # Preprocessed train, validation, and test splits
│       └── .gitkeep
│
├── models/
│   ├── .gitkeep
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF feature extractor
│   ├── logistic_regression/
│   │   ├── .gitkeep
│   │   └── model.joblib               # Trained Logistic Regression model
│   ├── svm/
│   │   ├── .gitkeep
│   │   └── model.joblib               # Trained Linear SVM model
│   └── distilbert/
│       ├── .gitkeep
│       ├── config.json                # Fine-tuned DistilBERT configuration
│       ├── model.safetensors          # Fine-tuned PyTorch model weights
│       └── tokenizer/                 # DistilBERT tokenizer vocabulary and assets
│
├── database/
│   ├── .gitkeep
│   └── predictions.db                 # SQLite database storing prediction history
│
├── reports/
│   ├── .gitkeep
│   └── figures/                       # Generated evaluation plots and visualizations
│
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── data_loader.py                 # Hugging Face dataset download & caching
│   ├── preprocess.py                  # Text cleaning, normalization & tokenization
│   ├── train_ml.py                    # Classical ML training (TF-IDF + LR / SVM)
│   ├── train_distilbert.py            # DistilBERT fine-tuning pipeline
│   ├── predict.py                     # Unified inference engine across all models
│   └── database.py                    # SQLite connection & schema management
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis & visual profiling
│   └── 02_model_evaluation.ipynb      # Comparative metrics, confusion matrices, ROC curves
│
├── static/
│   ├── css/
│   │   └── style.css                  # Custom styling and animations
│   └── js/
│       └── app.js                     # Async API calls and DOM interactions
│
├── templates/
│   ├── base.html                      # Bootstrap base layout
│   ├── index.html                     # Real-time text analysis interface
│   └── history.html                   # Prediction audit table and filters
│
├── app.py                             # Flask web application entrypoint
├── config.py                          # Global paths, constants, and hyperparameters
├── requirements.txt                   # Pinned dependency manifest
└── README.md                          # Project documentation
```

---

## 📊 Dataset Information

- **Dataset Source**: [`cike-dev/cyberbullying_dataset`](https://huggingface.co/datasets/cike-dev/cyberbullying_dataset) on Hugging Face.
- **Dataset Size**: ~30,000 annotated social media text samples.
- **Task**: Binary Classification (`bully` vs. `normal`).
- **Standardized Mapping**:
  - `0` ➔ `Non-Bullying`
  - `1` ➔ `Bullying`
- **Characteristics**: Diverse real-world social media vocabulary, informal slang, abbreviations, hashtags, user mentions, and emoji usage.

---

## ⚙️ Installation & Setup

### 1. Clone Repository & Navigate
```bash
git clone https://github.com/your-username/cyberbullying-detection.git
cd cyberbullying-detection
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

### 1. Exploratory Data Analysis
Launch Jupyter Notebook to inspect dataset balance, word frequency distributions, and n-grams:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train Machine Learning Models (TF-IDF + LR / SVM)
Trains the TF-IDF vectorizer, fits Logistic Regression and Linear SVM, and exports artifacts to `models/`:
```bash
python src/train_ml.py
```

### 3. Fine-Tune DistilBERT
Fine-tunes the transformer model on the processed dataset and saves checkpoints to `models/distilbert/`:
```bash
python src/train_distilbert.py
```

### 4. Run the Flask Web Application
Starts the local development server at `http://127.0.0.1:5000/`:
```bash
python app.py
```

---

## 📈 Model Performance Benchmark

The models are evaluated on a held-out test split (15%) across standard classification metrics:

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Inference Latency (CPU) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | *Pending* | *Pending* | *Pending* | *Pending* | ~2 ms |
| **TF-IDF + Linear SVM** | *Pending* | *Pending* | *Pending* | *Pending* | ~2 ms |
| **Fine-tuned DistilBERT** | *Pending* | *Pending* | *Pending* | *Pending* | ~35 ms |

> *Note: Populate the exact values upon running `python src/train_ml.py` and `python src/train_distilbert.py` or executing `notebooks/02_model_evaluation.ipynb`.*

---

## ⚖️ Ethical Considerations

1. **Content Sensitivity**: The underlying dataset contains toxic, offensive, and emotionally triggering text. Researchers and evaluators should exercise caution when inspecting raw data.
2. **False Positives / False Negatives**:
   - A *false positive* can censor benign humor, reclaimed language, or legitimate discourse.
   - A *false negative* allows harmful abuse to persist, endangering user mental health.
3. **Algorithmic Bias**: Social media models can inherit demographic, dialectal (e.g., AAVE), or cultural biases present in training annotations.
4. **Human-in-the-Loop Moderation**: Automated classification should assist human moderators rather than function as an unchecked autonomous banning authority.

---

## ⚠️ Limitations

- **Sarcasm & Subtlety**: Implicit hostility, passive-aggression, and sarcastic commentary remain difficult to detect without conversational context.
- **Linguistic Coverage**: The primary model is trained on English-language corpora; performance on code-mixed, multilingual, or non-English text is degraded.
- **Domain Shift**: Social media vernacular evolves rapidly (new slang, obfuscated spellings, acronyms), requiring continuous retraining.
- **Multimodal Absence**: Bullying frequently combines imagery, audio, and video (memes, reaction clips), which pure text models cannot parse.

---

## 🔮 Future Work

- [ ] **Multimodal Toxicity Analysis**: Integrating vision-language transformers (e.g., CLIP) for meme and image-text moderation.
- [ ] **Fine-Grained Hate Speech Taxonomy**: Expanding from binary detection into multi-label categories (cyberstalking, racism, sexism, physical threats).
- [ ] **Explainability & Interpretability**: Incorporating SHAP and Integrated Gradients to highlight exact trigger phrases in the UI.
- [ ] **Edge Deployment**: Quantizing DistilBERT with ONNX Runtime for low-latency browser or edge-device execution.
- [ ] **Active Learning Loop**: Allowing moderators to flag misclassifications to augment the retraining dataset.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for open-source model repositories and the `datasets` library.
- [cike-dev](https://huggingface.co/datasets/cike-dev/cyberbullying_dataset) for curating and hosting the cyberbullying dataset.
- The PyTorch, Scikit-Learn, and Flask open-source developer communities.
- Sanh et al. (2019) for the [DistilBERT architecture](https://arxiv.org/abs/1910.01108).
