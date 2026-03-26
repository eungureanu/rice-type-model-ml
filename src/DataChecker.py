import pandas as pd
import numpy as np

class DatasetLoader:

    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath
        self.df = None

    # incarcam fisierul intr-un DataFrame
    def load_data(self):
        self.df = pd.read_csv(self.dataset_filepath)
        return self.df

class DatasetReader:
    def __init__(self, df):
        #lucram cu o copie, ca sa pastram si datasetul original
        self.df = df.copy()

    def overview(self):
        print("Shape:", self.df.shape)
        print("\nColoane:", self.df.columns.tolist())
        print("\nData types:\n", self.df.dtypes)
        print("\nSumar:\n", self.df.describe())
        #intrebare: pastram prima coloana??

    def missing_values(self):
        print("\nValori lipsa:")
        print(self.df.isnull().sum())

    """
        test doc - why not working?
    """
    def outliers(self):
        print("\nOutlieri (metoda IQR):")
        # IQR = Interquartile Range
        # identifica outlieri relativ la dispersia mijlocului datelor (the middle 50% of the data)
        # Q1 (25%) first quartile, (75%) → third quartile; IQR=Q3−Q1
        # o valoare este outlier daca este <Q1−1.5*IQR sau > Q3+1.5*IQR

        # returneaza coloanele cu datatype numeric (int, float etc)
        valori_numerice = self.df.select_dtypes(include=np.number)

        for coloana in valori_numerice.columns:
            q1 = valori_numerice[coloana].quantile(0.25)
            q3 = valori_numerice[coloana].quantile(0.75)
            iqr = q3 - q1

            lower_q = q1 - 1.5 * iqr
            upper_q = q3 + 1.5 * iqr

            outliers = valori_numerice[(valori_numerice[coloana] < lower_q) | (valori_numerice[coloana] > upper_q)]
            print(f"{coloana}: {len(outliers)} outliers")


    def split_features_target(self):
        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        print("\nX shape:", X.shape)
        print("y shape:", y.shape)

        return X, y
