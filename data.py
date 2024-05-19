import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


class Data:
    def __init__(self, model_name, train_path, test_path):
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        self.model_name = model_name

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        # thêm padding để các câu có độ dài bằng nhau
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        return tokenizer

    def label_encode(self, train_dataset, test_dataset):
        label2id = {value: index for index, value in enumerate(
            self.df_train["labels"].unique())}
        id2label = {v: k for k, v in label2id.items()}

        def encode_labels(example):
            example["labels"] = label2id[example["labels"]]
            return example
        train_dataset = train_dataset.map(encode_labels)
        test_dataset = test_dataset.map(encode_labels)
        return train_dataset, test_dataset,  label2id, id2label

    def df_to_dataset(self, df):
        return Dataset.from_pandas(df).shuffle(seed=20)

    def tokenize(self, tokenizer, train_dataset, test_dataset):
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns='text')
        test_dataset = test_dataset.map(
            tokenize_function, batched=True, remove_columns='text')
        return train_dataset, test_dataset

    def process_dataset(self):
        tokenizer = self.get_tokenizer()
        train_dataset = self.df_to_dataset(self.df_train)
        test_dataset = self.df_to_dataset(self.df_test)

        train_dataset, test_dataset = self.tokenize(
            tokenizer, train_dataset, test_dataset)
        train_dataset, test_dataset, label2id, id2label = self.label_encode(
            train_dataset, test_dataset)

        return train_dataset, test_dataset, label2id, id2label
