import DataChecker as DataChecker

# Load dataset
data_loader = DataChecker.DatasetLoader("/Users/Emma/Uni/An III/ML/Rice_data_type.csv")
df = data_loader.load_data()

data_reader = DataChecker.DatasetReader(df)
overview = data_reader.overview()
valori_lipsa = data_reader.missing_values()
outliers = data_reader.outliers()
X, y = data_reader.split_features_target()
