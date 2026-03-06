import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def preprocess_data(data_file: str, output_dir: str) -> None:
    """
    Preprocess raw protein sequence data for model training.

    This function loads the raw data, cleans it, encodes labels, and splits
    it into train/validation/test sets. The split strategy must handle the
    extreme class imbalance in the Pfam dataset.

    Parameters
    ----------
    data_file : str
        Path to the combined raw data CSV file.
    output_dir : str
        Directory where processed files will be saved.

    Steps
    -----
    1. Load the data with pandas
    2. Remove rows with missing values
    3. Encode the 'family_accession' column with LabelEncoder
    4. Design and implement a split strategy that handles class imbalance
    5. Save train.csv, val.csv, and test.csv to output_dir

    Notes
    -----
    sklearn's train_test_split with stratify will fail on this dataset
    because some classes have only one sample. You need to implement
    a custom strategy.
    """
    data_path = Path(data_file)
    output_path = Path(output_dir)

    # TODO: implement preprocessing logic
    df = pd.read_csv(data_path)
    df = df.dropna()
    le = LabelEncoder()
    df['family_accession'] = le.fit_transform(df['family_accession'])
    class_counts = df["family_accession"].value_counts()

    classes_1 = class_counts[class_counts == 1].index
    classes_2 = class_counts[class_counts == 2].index
    classes_3plus = class_counts[class_counts >= 3].index

    df_1 = df[df["family_accession"].isin(classes_1)]

    df_2 = df[df["family_accession"].isin(classes_2)]

    train_2 = []
    test_2 = []

    for label in classes_2:
        samples = df_2[df_2["family_accession"] == label]
        train_2.append(samples.iloc[0])
        test_2.append(samples.iloc[1])

    train_2 = pd.DataFrame(train_2)
    test_2 = pd.DataFrame(test_2)

    df_3 = df[df["family_accession"].isin(classes_3plus)]

    train_tmp, test_3 = train_test_split(
        df_3,
        test_size=0.15,
        stratify=df_3["family_accession"],
        random_state=42,
    )

    train_3, val_3 = train_test_split(
        train_tmp,
        test_size=0.15,
        stratify=train_tmp["family_accession"],
        random_state=42,
    )

    train_df = pd.concat([df_1, train_2, train_3], ignore_index=True)
    val_df = val_3
    test_df = pd.concat([test_2, test_3], ignore_index=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Pfam data.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
