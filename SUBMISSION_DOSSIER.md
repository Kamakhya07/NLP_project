# 🎓 Final-Year NLP Project Submission Dossier
## Evaluation Criteria Mapping, Mandatory Deliverables & Submission Form

---

## Part 1: Project Submission Form (Ready to Copy-Paste)

| Information Field | Submission Details |
| :--- | :--- |
| **Project Title** | **CyberShield: Contextual NLP and Transformer-Based Cyberbullying Detection from Social Media** |
| **Student Name(s)** | Kamakhya Kanzade |
| **PRN / Roll Number(s)** | *[Enter your PRN / Roll Number here]* |
| **Assigned Problem Statement** | Design, implement, and evaluate a contextual Natural Language Processing (NLP) text classification system to detect cyberbullying and abusive language in social media comments, comparing lexical n-gram baselines against deep contextual Transformers, and deploy as an interactive web solution with Generative AI counter-speech. |
| **GitHub Repository Link** | **[https://github.com/Kamakhya07/NLP_project](https://github.com/Kamakhya07/NLP_project)** |
| **Deployed Project Link** | **Live Local URL:** `http://127.0.0.1:5000/`<br/>*(For cloud deployment instructions via Render / Hugging Face Spaces, see Section 3 below)* |
| **Project Poster** | **Interactive Printable Poster:** `http://127.0.0.1:5000/poster`<br/>*(Source template: `templates/poster.html`)* |
| **Output Screenshots** | Stored in repository at `reports/figures/`<br/>- Confusion Matrix (Logistic Regression): `reports/figures/confusion_matrix_logistic_regression.png`<br/>- Confusion Matrix (Linear SVM): `reports/figures/confusion_matrix_svm.png`<br/>- Benchmark Comparison Chart: `reports/figures/model_comparison_chart.png` |
| **Dataset Details** | **CardiffNLP TweetEval Offensive Benchmark** augmented with balanced conversational expressions (~14,230 social media comments; 75% Train, 12.5% Validation, 12.5% Test). |
| **AWS Certificate** | Downloaded NLP Lab completion certificate (Attach PDF in your submission folder). |
| **Shared Folder Link** | *[Insert Google Drive / OneDrive Shared Folder Link]* |

---

## Part 2: Demonstration Against Evaluation Criteria (Rubric Points 2 to 7)

### Criterion 2: Text Extraction, Preprocessing & Vectorization
- **Implementation:** [src/preprocessing.py](file:///D:/TY/NLP-project/src/preprocessing.py) and [notebooks/02_preprocessing.ipynb](file:///D:/TY/NLP-project/notebooks/02_preprocessing.ipynb).
- **Techniques Used:**
  1. *Conservative Normalization:* Stripping noisy URLs (`http://...`) and platform user handles (`@user`) while deliberately preserving expressive punctuation (`!`, `?`) and emojis to retain affective tone.
  2. *N-Gram Vectorization:* Scikit-Learn `TfidfVectorizer` extracting word unigrams, bigrams, and trigrams (`ngram_range=(1, 3)`, `min_df=2`, `max_features=50000`, `sublinear_tf=True`) producing **39,256 linguistic features**.
  3. *Subword Tokenization:* DistilBERT WordPiece tokenizer (`max_length=128`, padding & truncation).

### Criterion 3: Implementation of NLP Algorithms
- **Implementation:** [src/train_ml.py](file:///D:/TY/NLP-project/src/train_ml.py) and [src/train_distilbert.py](file:///D:/TY/NLP-project/src/train_distilbert.py).
- **Algorithms:**
  1. *Logistic Regression:* $L_2$-regularized linear model with balanced class weighting.
  2. *Linear SVM:* Max-margin classification wrapped in `CalibratedClassifierCV` via Platt Scaling sigmoid calibration for reliable probability estimation.
  3. *DistilBERT Transformer:* 6-layer bidirectional Transformer encoder with 12 multi-head self-attention mechanisms and a sequence classification head.

### Criterion 4: Working Solution for Sentiment / Toxicity Classification
- **Implementation:** Interactive Flask Web Application ([app.py](file:///D:/TY/NLP-project/app.py)) with SQLite audit logging ([templates/history.html](file:///D:/TY/NLP-project/templates/history.html)) and dynamic Chart.js analytics dashboard ([templates/dashboard.html](file:///D:/TY/NLP-project/templates/dashboard.html)).
- **Real-Time Model Switching:** Evaluates any entered social media comment and displays verdict badges, calibrated confidence meters, and class probabilities.

### Criterion 5: Cloud AI / Generative AI & Prompt Engineering
- **Implementation:** [src/genai_explainer.py](file:///D:/TY/NLP-project/src/genai_explainer.py) and embedded in [templates/result.html](file:///D:/TY/NLP-project/templates/result.html).
- **Prompt Engineering Concepts Demonstrated:**
  1. *Role Prompting:* Assigns the LLM a persona as an *"NLP Safety Auditor and Restorative Communication Specialist"*.
  2. *Few-Shot In-Context Learning:* Provides structured exemplars of abusive comments mapped to attack categories and constructive counter-speech.
  3. *Chain-of-Thought (CoT) Linguistic Reasoning:* Generates Explainable AI (XAI) rationale detailing *why* the comment was flagged.
  4. *Empathetic Counter-Speech & Non-Violent Rephrasing:* Translates hostile attacks into civil, non-violent expressions.
  5. *View Engineered Prompt:* Accessible via an interactive accordion in the web UI.

### Criterion 6 & 7: Methodology Explanation & Technical Contributions
- **Methodology Handbook:** [project_report.md](file:///C:/Users/kamakhya/.gemini/antigravity/brain/427a81e8-851c-4696-8fdf-98977d784031/project_report.md).
- **Oral Defense & Demo Playbook:** [presentation_and_demo_guide.md](file:///C:/Users/kamakhya/.gemini/antigravity/brain/427a81e8-851c-4696-8fdf-98977d784031/presentation_and_demo_guide.md).
- **Key Technical Innovation:** Identified and eliminated **spurious lexical pronoun bias** (where harmless phrases like *"i like you"* were falsely penalized by naive bag-of-words models), achieving balanced conversational accuracy across trigram feature spaces.

---

## Part 3: Mandatory Deliverables Checklist

- [x] **1. Project Title & Problem Statement:** Documented in README and Dossier.
- [x] **2. GitHub Repository Link:** [https://github.com/Kamakhya07/NLP_project](https://github.com/Kamakhya07/NLP_project).
- [x] **3. README.md File:** Comprehensive documentation with architecture tree, installation, and usage.
- [x] **4. Deployed Project Link:**
  - Running locally at `http://127.0.0.1:5000/`.
  - To share a live public link during demo, run:
    ```bash
    # Option A: using localtunnel (free, no sign-up)
    npx localtunnel --port 5000
    # Option B: using ngrok
    ngrok http 5000
    ```
- [x] **5. Project Poster:** High-resolution printable one-page poster available at `http://127.0.0.1:5000/poster` (Click *"Print / Save as PDF"*).
- [x] **6. Output Screenshots:** High-res confusion matrices and comparison charts in `reports/figures/`.
- [x] **7. Dataset Details:** CardiffNLP TweetEval Offensive benchmark documented in `dataset/`.
- [x] **8. AWS Certificate:** Downloaded PDF certificate to be placed in shared folder.
