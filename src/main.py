import DataChecker as DataChecker

# Load dataset
data_loader = DataChecker.DatasetLoader("/Users/Emma/Uni/An III/ML/Rice_data_type.csv")
df = data_loader.load_data()

data_reader = DataChecker.DatasetReader(df)
info = data_reader.data_summary()
