import pandas as pd

class DatasetLoader:

    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath
        self.df = None

    def load_data(self):
        # incarcam fisierul intr-un DataFrame
        self.df = pd.read_csv(self.dataset_filepath)
        return self.df

class DatasetReader:
    def __init__(self, df):
        self.df = df

    def data_summary(self):
        print(f"Rice Dataset Shape {self.df.shape}\n")
        print(f"Rice Data Types {self.df.dtypes})\n")