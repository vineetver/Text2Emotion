from src.dataset.create_dataset import read_dataset, write_dataset, combine_dataset, split_dataset
from src.feature.preprocessing import drop_annotator_column, str_to_index, apply_index_to_class, apply_ekman_mapping, \
    apply_clean_text, one_hot_encode

train_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/train.tsv'
valid_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/dev.tsv'
test_url = 'https://github.com/google-research/google-research/raw/master/goemotions/data/test.tsv'


def main():

    # Read the training, testing, and validation datasets
    train_df, test_df, valid_df = read_dataset(train_url, test_url, valid_url)

    # Combine the training, testing, and validation datasets using concat
    df = combine_dataset(train_df, test_df, valid_df)

    # drop the 'annotator' column from the dataset
    df = drop_annotator_column(df)

    # convert the string class indices to list of indices
    df = str_to_index(df)

    # convert the list of indices to list of class labels
    df = apply_index_to_class(df)

    # apply the Ekman mapping to the dataset to reduce the number of classes
    df = apply_ekman_mapping(df)

    # apply the clean text function to the dataset (remove punctuation, lowercase, etc.)
    df = apply_clean_text(df)

    # one-hot encode the class labels
    df = one_hot_encode(df)

    # split the dataset into training, testing, and validation datasets
    train_df, test_df, valid_df = split_dataset(df)

    # write the training, testing, and validation datasets to '/data'
    write_dataset(train_df, valid_df, test_df)


if __name__ == "__main__":
    main()
