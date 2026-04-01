"""
Configurare centrala pentru a ne asigura ca rezultatele experimentelor rulate cu aceleasi setari genereaza aceleasi rezultate.
"""

from pathlib import Path

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET_COLUMN = "Class"
DATASET_PATH = "/Users/Emma/Uni/An III/ML/rice_data_type.csv"
DEFAULT_TEST_SIZE = 0.2

# Cerinta: rulați experimente cu minimum 3 variante de împărțire, modificând cel puțin unul dintre următoarele: proporția (ex. 60/40, 70/30, 80/20), stratificarea pe clase (acolo unde are sens), valoarea seed-ului (minimum 3 valori).
VARIANTE_IMPARTIRE = [
    {"test_size": 0.4, "random_state": SEED, "stratify": True},
    {"test_size": 0.3, "random_state": SEED, "stratify": True},
    {"test_size": 0.2, "random_state": SEED, "stratify": True},
]
# Alternativa: acelasi test_size, cu diferite seeds
# VARIANTE_IMPARTIRE = [
#     {"test_size": DEFAULT_TEST_SIZE, "random_state": 42, "stratify": True},
#     {"test_size": DEFAULT_TEST_SIZE, "random_state": 123, "stratify": True},
#     {"test_size": DEFAULT_TEST_SIZE, "random_state": 456, "stratify": True},
# ]
