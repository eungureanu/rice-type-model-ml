from sklearn.model_selection import train_test_split
import config

def split_features_target(df):
    """X = atribute; y = etichete decodate."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y

def split_train_test(X, y, test_size, random_state):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
