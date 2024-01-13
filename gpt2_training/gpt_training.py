from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from transformers.utils import logging
import numpy as np
from gpt2_training.config import parser

import torch

import pandas as pd

args = parser.parse_args()

logging.set_verbosity_info()
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
logger = logging.get_logger("transformers")


# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.modelname, pad_token="<|pad|>")
tokenizer.pad_token_id = tokenizer.eos_token_id


# same as the bart we used 20 90% query
def trunction_query(row):
    return " ".join(row.split()[: args.input_length])


def trunction_target(row):
    return " ".join(row.split()[: args.target_length])


def calculate_rouge(output_lns, reference_lns):
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return result


class S2Sdataset(Dataset):
    def __init__(self, max_length=20, data_type="train"):
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


train_dataset = S2Sdataset(max_length=args.max_sequenge_length, data_type="train")
valid_dataset = S2Sdataset(max_length=args.max_sequenge_length, data_type="val")
# test_dataset = S2Sdataset(max_length=200, data_type="test")


training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epoch,
    logging_steps=10,
    load_best_model_at_end=False,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="logs",
    save_total_limit=1,  # only save one file
    # report_to="none",
    # metric_for_best_model="rouge",
)

model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
model.resize_token_embeddings(len(tokenizer))


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()


def predict():
    fout = open(f"{args.predict_data_path}", "w")
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # generated_length = 20
    # target_length = 12
    # test_source_path = "scifact/test.source"

    generated_length = 20
    target_length = 20
    test_source_path = "data/nq-answer/test.source"

    df = pd.read_csv(test_source_path, names=["source"], sep="\t")
    for text in tqdm(df["source"].to_list()):
        prompt = text

        # note that we only want to make the parameters same as the bart, so we set the max_length of query to 20.
        generated = tokenizer(
            f"{prompt}", return_tensors="pt", max_length=generated_length
        ).input_ids.cuda()

        # perform prediction
        sample_outputs = model.generate(
            generated,
            do_sample=False,
            top_k=50,
            max_length=40,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        predicted_text = predicted_text.strip().replace("\n", "")
        target = " ".join(
            predicted_text.split("<|pad|>")[-1].split(" ")[:target_length]
        )
        fout.write(target + "\n")
        fout.flush()
        # print("predicted_text", predicted_text)


predict()

if __name__ == "__main__":
    train(args)
