import pandas as pd

DATA_PATH = "data/data.csv"

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    dataset = load_dataset()