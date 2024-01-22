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
    def __init__(self, args, tokenizer=None, data_type="train"):
        self.input_ids = []
        self.attn_masks = []

        assert tokenizer is not None

        source_items = open(f"{args.data_path}/{data_type}.source", "r").readlines()
        self.df_source = pd.DataFrame(source_items, columns=["source"])

        target_items = open(f"{args.data_path}/{data_type}.target", "r").readlines()
        self.df_target = pd.DataFrame(target_items, columns=["target"])
        self.df_source["source"] = self.df_source["source"].apply(
            trunction_query, input_length=args.input_length
        )
        self.df_target["target"] = self.df_target["target"].apply(
            trunction_target, target_length=args.target_length
        )

        df_merge = self.df_source["source"] + "<|sep|>" + self.df_target["target"]

        for item in df_merge:
            # tokenize
            encodings_dict = tokenizer(
                item,
                truncation=True,
                max_length=args.max_sequence_length,
                padding="max_length",
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
