"""Text Preprocessing Module for Cyberbullying Detection (Compatibility Layer).

Exposes `clean_text` and `preprocess_text` by importing from `src.preprocessing`.
Ensures backward compatibility across all modules.
"""

from src.preprocessing import clean_text, preprocess_text, preprocess_dataframe

__all__ = ["clean_text", "preprocess_text", "preprocess_dataframe"]

if __name__ == "__main__":
    sample = "Hey @johndoe check this out https://example.com #CyberBullying is BAD!!!! 😡"
    print("Cleaned:", clean_text(sample))
