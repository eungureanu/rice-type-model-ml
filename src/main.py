"""Entry point: incarcare dataset si afisare informatii."""

from data_loading import (
    DatasetLoader,
    EDAReport,
    print_class_distribution,
    split_features_target,
)


def main():
    df = DatasetLoader().load()
    print("Shape:", df.shape)
    X, y = split_features_target(df)
    print("Features:", list(X.columns))
    print("Classes:", sorted(y.unique().tolist()))
    print_class_distribution(y)
    print()
    EDAReport(df).print_report()


if __name__ == "__main__":
    main()
