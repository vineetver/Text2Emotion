from src.dataset.create_dataset import read_dataset, write_dataset, combine_dataset, split_dataset
from src.feature.preprocessing import drop_annotator_column, str_to_index

train_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/train.tsv'
valid_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/dev.tsv'
test_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/test.tsv'


def main():
    train_df, test_df, valid_df = read_dataset(train_url, test_url, valid_url)

    df = combine_dataset(train_df, test_df, valid_df)

    df = drop_annotator_column(df)

    df = str_to_index(df)

    print(train_df.head())

    train_df, test_df, valid_df = split_dataset(df)

    write_dataset(train_df, valid_df, test_df)


if __name__ == "__main__":
    main()
