import pytorch_lightning as pl
from data_util import config
from torch.utils.data import random_split, DataLoader
from data_util.data import Vocab
from data_util.batcher import Batcher
import pandas as pd
from torch.utils.data import Dataset

vocab = Vocab(config.vocab_path, config.vocab_size)

print(vocab)
batch = Batcher(
    config.train_data_path,
    vocab,
    mode="train",
    batch_size=config.batch_size,
    single_pass=False,
)


class S2SDataset(Dataset):
    def __init__(
        self,
        type="train",
        vocab=vocab,
    ):
        self.df_source = pd.read_csv(
            f"scifact/{type}.source", sep="\t", names=["source"]
        )
        self.df_target = pd.read_csv(
            f"scifact/{type}.target", sep="\t", names=["target"]
        )
    def 
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def texts2ids(self, target, source):
        # Process the source
        source_words = source.split()
        if len(source_words) > config.max_enc_steps:
            source_words = source_words[: config.max_enc_steps]
        self.enc_len = len(
            source_words
        )  # store the length after truncation but before padding
        self.enc_input = [
            vocab.word2id(w) for w in source_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the target
        target_words = target.split()  # list of strings
        abs_ids = [
            vocab.word2id(w) for w in target_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

    def __len__(self):
        return len(self.df_target)

    def __getitem__(self, index):
        return {
            "target": self.df_target["target"].iloc[index],
            "source": self.df_source["source"].iloc[index],
        }


class S2SDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path") -> None:
        super().__init__()

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.train_data = S2SDataset("train", self.vocab)
        self.valid_data = S2SDataset("valid", self.vocab)
        self.batch_size = 64

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)
