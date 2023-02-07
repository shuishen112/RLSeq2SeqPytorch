import pytorch_lightning as pl
from data_util import config
from data_util import data
import numpy as np

from torch.utils.data import random_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")


def yield_tokens(df_list):
    for df in df_list:
        for text in df:
            yield tokenizer(text)


class Example(object):
    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.__getitem__(data.START_DECODING)
        stop_decoding = vocab.__getitem__(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[: config.max_enc_steps]
        self.enc_len = len(
            article_words
        )  # store the length after truncation but before padding
        self.enc_input = [
            vocab.__getitem__(w) for w in article_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = " ".join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [
            vocab.__getitem__(w) for w in abstract_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, _ = self.get_dec_inp_targ_seqs(
            abs_ids, config.max_dec_steps, start_decoding, stop_decoding
        )
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
        self.enc_input_extend_vocab, self.article_oovs = data.article2ids_new(
            article_words, vocab
        )

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids_new(
            abstract_words, vocab, self.article_oovs
        )

        # Get decoder target sequence
        _, self.target = self.get_dec_inp_targ_seqs(
            abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding
        )

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

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

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)


class S2SDataset(Dataset):
    def __init__(
        self,
        vocab,
        type="train",
    ):
        """_summary_

        Args:
            vocab (_type_): vocab
            type (_type_): train, test, valid
        """
        self.df_article = pd.read_csv(
            f"scifact/{type}.source", sep="\t", names=["article"]
        )
        self.df_abstract = pd.read_csv(
            f"scifact/{type}.target", sep="\t", names=["abstract"]
        )
        # self.df_article = pd.read_csv(
        #     f"data/unfinished/{type}.art.shuf1000.txt", sep="\t", names=["article"]
        # )

        # self.df_abstract = pd.read_csv(
        #     f"data/unfinished/{type}.abs.shuf1000.txt", sep="\t", names=["abstract"]
        # )

        self.vocab = vocab

    def __len__(self):
        return len(self.df_article)

    def __getitem__(self, index):
        article = self.df_article["article"].iloc[index]
        abstract = self.df_abstract["abstract"].iloc[index]
        ex = Example(article, [abstract], self.vocab)
        return ex


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.__getitem__(
            data.PAD_TOKEN
        )  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(
            example_list
        )  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros(
            (self.batch_size, max_enc_seq_len), dtype=np.float32
        )

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros(
            (self.batch_size, max_enc_seq_len), dtype=np.int32
        )
        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros(
            (self.batch_size, config.max_dec_steps), dtype=np.int32
        )
        self.target_batch = np.zeros(
            (self.batch_size, config.max_dec_steps), dtype=np.int32
        )
        # self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            # for j in range(ex.dec_len):
            #   self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [
            ex.original_article for ex in example_list
        ]  # list of lists
        self.original_abstracts = [
            ex.original_abstract for ex in example_list
        ]  # list of lists
        self.original_abstracts_sents = [
            ex.original_abstract_sents for ex in example_list
        ]  # list of list of lists


class S2SDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path") -> None:
        super().__init__()

        # df_article = pd.read_csv(
        #     "data/unfinished/train.art.shuf1000.txt", sep="\t", names=["article"]
        # )
        # df_abstract = pd.read_csv(
        #     "data/unfinished/train.abs.shuf1000.txt", sep="\t", names=["abstract"]
        # )

        df_article = pd.read_csv("scifact/train.source", sep="\t", names=["article"])
        df_abstract = pd.read_csv("scifact/train.target", sep="\t", names=["abstract"])

        self.batch_size = config.batch_size
        # PAD_TOKEN = "[PAD]"  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        # UNKNOWN_TOKEN = "[UNK]"  # This has a vocab id, which is used to represent out-of-vocabulary words
        # START_DECODING = "[START]"  # This has a vocab id, which is used at the start of every decoder input sequence
        # STOP_DECODING = "[STOP]"  # This has a vocab id, which is used at the end of untruncated target sequences

        # build the vocabulary
        self.vocab = build_vocab_from_iterator(
            yield_tokens(
                [df_abstract["abstract"].to_list(), df_article["article"].to_list()]
            ),
            specials=["[UNK]", "[PAD]", "[START]", "[STOP]"],
        )
        self.vocab.set_default_index(self.vocab["[UNK]"])

        # get the dataset

        self.train_dataset = S2SDataset(self.vocab, "train")
        self.valid_dataset = S2SDataset(self.vocab, "val")
        self.test_dataset = S2SDataset(self.vocab, "test")

        # self.train_dataset = S2SDataset(self.vocab, "train")
        # self.valid_dataset = S2SDataset(self.vocab, "valid")

    def collate_batch(self, batch):
        return Batch(batch, self.vocab, len(batch))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_batch,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=self.collate_batch,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_batch,
            batch_size=50,
        )
