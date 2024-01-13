import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm
import torch


# same as the bart we used 20 90% query
def trunction_query(row, input_length=20):
    return " ".join(row.split()[:input_length])


def trunction_target(row, target_length=20):
    return " ".join(row.split()[:target_length])


class S2Sdataset(Dataset):
    def __init__(self, args, tokenizer=None, max_length=20, data_type="train"):
        self.input_ids = []
        self.attn_masks = []

        self.df_source = pd.read_csv(
            f"{args.data_path}/{data_type}.source", sep="\t", names=["source"]
        )
        self.df_target = pd.read_csv(
            f"{args.data_path}/{data_type}.target", sep="\t", names=["target"]
        )
        self.df_source["source"] = self.df_source["source"].apply(trunction_query)
        self.df_target["target"] = self.df_target["target"].apply(trunction_target)

        df_merge = self.df_source["source"] + "<|pad|>" + self.df_target["target"]

        for item in tqdm(df_merge):
            # tokenize
            encodings_dict = tokenizer(
                item, truncation=True, max_length=max_length, padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attn_masks[index],
            "labels": self.input_ids[index],
            "source": self.df_source["source"].iloc[index],
            "target": self.df_target["target"].iloc[index],
        }
