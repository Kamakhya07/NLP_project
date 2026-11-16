"""
CyberShield: Baseline Model Trainer
Trains initial TF-IDF vectorizer, Logistic Regression, and Linear SVM models
and saves them to the required project paths.
"""

import os
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.preprocess import clean_text
except ModuleNotFoundError:
    from preprocess import clean_text

# Seed data representing typical cyberbullying and benign social media comments
SAMPLE_DATA = [
    # Bullying / Toxic (label 1)
    ("You are so pathetic and ugly, nobody likes you. Delete your account and disappear.", 1),
    ("Keep talking and you will seriously regret showing your face anywhere near here, loser.", 1),
    ("Nobody wants you here, go kill yourself freak.", 1),
    ("You are an idiot and your family hates you.", 1),
    ("Shut up you worthless scum, you are pure trash.", 1),
    ("Disgusting retard, nobody asked for your stupid opinion.", 1),
    ("Go die in a ditch, literally everyone despises you.", 1),
    ("Look at how fat and ugly you are, what a joke of a person.", 1),
    ("You deserve all the pain coming your way, creep.", 1),
    ("Get off the internet loser, nobody cares about you.", 1),
    ("You're a disgusting piece of garbage and should hang yourself.", 1),
    ("Shut up bitch, go cry somewhere else.", 1),
    ("You are completely brain dead and worthless.", 1),
    ("Drop dead idiot, nobody loves you.", 1),
    ("I hope you suffer every single day for being so vile.", 1),
    ("You are a hopeless clown, stop embarrassing yourself.", 1),
    ("Nobody cares about your existence, why don't you just disappear.", 1),
    ("You are so stupid it actually hurts to listen to you.", 1),

    # Non-Bullying / Benign (label 0)
    ("Had an amazing time collaborating with our study group today! So grateful for supportive friends.", 0),
    ("I respectfully disagree with the points made in your review, but I appreciate your perspective.", 0),
    ("Does anyone know what time the library closes on Sundays?", 0),
    ("Happy birthday! Wishing you a fantastic year filled with joy and success.", 0),
    ("Great job on the presentation today, you explained the difficult concepts really clearly.", 0),
    ("Could you please share the link to the open source repository mentioned earlier?", 0),
    ("I loved the movie soundtrack, especially the violin soloist in the third act.", 0),
    ("Thanks for helping me fix that bug in my code, really appreciate your patience!", 0),
    ("The weather today is so nice, going out for a walk in the park.", 0),
    ("Let us schedule our project sync meeting for tomorrow at 2 PM.", 0),
    ("Congratulations on passing your exams! Proud of all your hard work.", 0),
    ("Interesting discussion on artificial intelligence ethics, thanks for organizing.", 0),
    ("Can someone recommend a good book on natural language processing?", 0),
    ("Looking forward to the weekend and spending time with family.", 0),
    ("That recipe was delicious, my friends enjoyed it a lot.", 0),
    ("You are an amazing and kind person, thank you so much for your help!", 0),
    ("You are awesome! Really appreciate everything you do for the community.", 0),
    ("You are a wonderful friend and I am so lucky to know you.", 0),
    ("You are very talented and creative, keep up the great work!", 0),
    ("Good morning everyone! Hope you all have a productive and peaceful day.", 0),
    ("What are your favorite podcasts to listen to while studying or commuting?", 0),
    ("Thank you for sharing this helpful guide, it saved me hours of debugging.", 0),
]


def train_and_save_baseline_models():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    lr_dir = models_dir / "logistic_regression"
    svm_dir = models_dir / "svm"

    lr_dir.mkdir(parents=True, exist_ok=True)
    svm_dir.mkdir(parents=True, exist_ok=True)

    texts = [clean_text(item[0]) for item in SAMPLE_DATA]
    labels = [item[1] for item in SAMPLE_DATA]

    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X, labels)

    print("Training Linear SVM...")
    svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm_model.fit(X, labels)

    # Save artifacts
    tfidf_path = models_dir / "tfidf_vectorizer.joblib"
    lr_path = lr_dir / "model.joblib"
    svm_path = svm_dir / "model.joblib"

    print(f"Saving vectorizer -> {tfidf_path}")
    joblib.dump(vectorizer, tfidf_path)

    print(f"Saving Logistic Regression -> {lr_path}")
    joblib.dump(lr_model, lr_path)

    print(f"Saving Linear SVM -> {svm_path}")
    joblib.dump(svm_model, svm_path)

    print("All baseline models successfully trained and serialized!")


if __name__ == "__main__":
    train_and_save_baseline_models()
