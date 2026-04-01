"""
Incarca datasetul, curata etichetele target si raporteaza distributia claselor.
Analizeaza si vizualizeaza datele (caracteristici principale, outlieri, patterns).
EDA = Exploratory Data Analysis
"""

import pandas as pd

import config

def decode_class_value(val):
    """Decodeaza coloana Class: scoate notatia b'...' din byte notation si lasa numele clasei ca string."""
    s = str(val).strip()
    # Verificam daca stringul este in formatul b'...' astfel:
        # len(s) >= 3
        # primul caracter este 'b' (Python’s “bytes literal” prefix)
        # al doilea caracter este un quote mark
        # ultimul caracter este acelasi cu al doilea caracter (quote mark)
    if len(s) >= 3 and s[0] == "b" and s[1] in "'\"" and s[-1] == s[1]:
        return s[2:-1]
    return s


def drop_leading_index_column(df):
    """Sterge prima coloana daca e index fara nume (Unnamed sau goala)."""
    if df.empty:
        return df
    first_column = df.columns[0]
    name = "" if first_column is None else str(first_column).strip()
    if name == "" or name.startswith("Unnamed"):
        return df.drop(columns=first_column)
    return df


class DatasetLoader:
    """
    Citeste CSV-ul, elimina coloana de index fara nume daca e cazul,
    decodeaza coloana tinta (Class). Foloseste calea din config sau una data explicit.
    """

    def __init__(self, csv_path=None):
        self.csv_path = csv_path if csv_path is not None else config.DATASET_PATH
        self.df = None

    def load(self):
        df = pd.read_csv(self.csv_path)
        df = drop_leading_index_column(df)
        df = df.copy()
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(decode_class_value)
        self.df = df
        return df


def load_dataset():
    """Wrapper: incarca folosind DatasetLoader cu calea din config."""
    return DatasetLoader().load()


def class_distribution_dataframe(y):
    """Tabel: count si procente pe clasa."""
    counts = y.value_counts()
    procent = (counts / len(y) * 100.0).round(2)
    out = pd.DataFrame({"count": counts, "procent": procent})
    out.index.name = "class"
    return out


def class_distribution_summary(dist):
    """O propozitie despre dezechilibru, pe baza tabelului."""
    if dist.empty or len(dist) < 2:
        return "Prea putine clase pentru o analiza de dezechilibru."

    procents = dist["procent"].astype(float)
    ratio = float(procents.max() / max(procents.min(), 1e-12))
    if ratio < 1.2:
        return (
            "Clasele sunt aproape echilibrate (raport max/min procente "
            "≈ %.2f)." % ratio
        )
    if ratio < 2.0:
        return (
            "Usor dezechilibru intre clase (raport max/min procente "
            "≈ %.2f)." % ratio
        )
    return (
        "Dezechilibru notabil intre clase (raport max/min procente "
        "≈ %.2f)."
        % ratio
    )


def class_distribution_report(y):
    """Intoarce tabelul si propozitia de dezechilibru."""
    dist = class_distribution_dataframe(y)
    return dist, class_distribution_summary(dist)


def split_features_target(df):
    """X = atribute; y = etichete decodate."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y


def print_class_distribution(y):
    dist, summary = class_distribution_report(y)
    print("Distribuție clase (count, %):")
    print(dist.to_string())
    print()
    print(summary)


def missing_values_report(df):
    """
    Count si procent date lipsa pe coloana
    """
    n = len(df)
    counts = df.isna().sum()
    percentage = (counts / n * 100.0).round(2) if n else counts * 0.0
    out = pd.DataFrame({"missing_count": counts, "missing_percentage": percentage})
    out.index.name = "column"
    return out


def feature_dtypes_summary(df):
    """Tabel: coloana, datatypes, numar de valori (non-nule)."""
    out = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.notna().sum(),
        }
    )
    out.index.name = "column"
    return out


def iqr_outliers_report(df, exclude_columns=None):
    """
    # IQR = InterQuartile Range
        # identifica outlieri relativ la dispersia mijlocului datelor (the middle 50% of the data)
        # Q1 (25%) first quartile, (75%) → third quartile; IQR=Q3−Q1
        # o valoare este outlier daca este <Q1−1.5*IQR sau > Q3+1.5*IQR
    """
    exclude = set(exclude_columns or [])
    numeric = df.select_dtypes(include="number")
    cols = [c for c in numeric.columns if c not in exclude]
    rows = []
    n_total = len(df)
    for col in cols:
        s = numeric[col].dropna()
        if s.empty:
            rows.append(
                {
                    "column": col,
                    "n_outliers": 0,
                    "outlier_percentage": 0.0,
                    "q1": float("nan"),
                    "q3": float("nan"),
                    "iqr": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                }
            )
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((numeric[col] < lower) | (numeric[col] > upper)).sum())
        percentage = round(100.0 * n_out / n_total, 2) if n_total else 0.0
        rows.append(
            {
                "column": col,
                "n_outliers": n_out,
                "outlier_percentage": percentage,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
    return pd.DataFrame(rows)


def global_descriptive_stats(df, exclude_columns=None):
    """
    Descrie setul de date.
    """
    exclude = set(exclude_columns or [])
    sub = df[[c for c in df.columns if c not in exclude]]
    num = sub.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    return num.describe()


class EDAReport:
    """
    Analiza exploratorie pe un DataFrame: tabele (missing, dtypes, IQR, describe)
    si afisare cu headline-uri.
    """

    def __init__(self, df):
        self.df = df.copy()

    def full_report(self):
        missing = missing_values_report(self.df)
        dtypes = feature_dtypes_summary(self.df)
        iqr = iqr_outliers_report(self.df, exclude_columns=[config.TARGET_COLUMN])
        describe = global_descriptive_stats(
            self.df, exclude_columns=[config.TARGET_COLUMN]
        )
        return {
            "missing_values": missing,
            "dtypes": dtypes,
            "iqr_outliers": iqr,
            "numeric_describe_global": describe,
        }

    def print_report(self):
        raport = self.full_report()
        numeric_describe = (
            "(niciun feature numeric)"
            if raport["numeric_describe_global"].empty
            else raport["numeric_describe_global"].to_string()
        )
        print(
            f"------ Tipuri de date ------\n{raport['dtypes'].to_string()}\n\n"
            f"------ Valori lipsa ------\n{raport['missing_values'].to_string()}\n\n"
            f"------ Outlieri IQR ------\n{raport['iqr_outliers'].to_string(index=False)}\n\n"
            f"------ Statistici descriptive ------\n{numeric_describe}\n"
        )
def full_report(df):
    """Dict EDA (aceeasi structura ca EDAReport.full_report)."""
    return EDAReport(df).full_report()


def print_report(df):
    """Afiseaza raportul EDA (delegat la EDAReport)."""
    EDAReport(df).print_report()
