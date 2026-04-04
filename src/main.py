from config import SEED, DEFAULT_TEST_SIZE
from data_loading import (
    DatasetLoader
)

from src.cerinte import *


def main():
    df = DatasetLoader().load()

    cerinta1_2_3_4(df)
    cerinta5(df)
    cerinta6(df, DEFAULT_TEST_SIZE, SEED)
    # cerinta7(df, DEFAULT_TEST_SIZE, SEED) #nu ruleaza din cauza diferentelor de versiune intre scikit-learn si imbalanced-learn
    cerinta8(df, DEFAULT_TEST_SIZE, SEED)
    cerinta9(df, DEFAULT_TEST_SIZE, SEED)
    cerinta10(df, DEFAULT_TEST_SIZE, SEED)
    cerinta11(df, DEFAULT_TEST_SIZE, SEED)
    cerinta12(df, DEFAULT_TEST_SIZE, SEED)


if __name__ == "__main__":
    main()
