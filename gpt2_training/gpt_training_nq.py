# %%
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from transformers.utils import logging
import numpy as np
from datasets import load_metric
import yaml

yaml_args = yaml.load(open("yaml_config/nq_gpt.yaml"), Loader=yaml.FullLoader)


logging.set_verbosity_info()
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
logger = logging.get_logger("transformers")


# %%
# load dataset
import pandas as pd

# %%
# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# data_path = "scifact"
# modelname = "gpt2"
# predict_data_path = "scifact/gpt_predict.txt"
# output_dir = "results"
# input_length = 20
# target_length = 40
# tokenizer_savedir = "./models/tokenizer/"
# epoch = 100
# max_sequenge_length = (
#     40  # gpt use the whole sentence to train, the max_sequence_length = source + target
# )

data_path = yaml_args["data_path"]
modelname = yaml_args["modelname"]
predict_data_path = yaml_args["predict_data_path"]
output_dir = yaml_args["output_dir"]
input_length = yaml_args["input_length"]
target_length = yaml_args["target_length"]
tokenizer_savedir = yaml_args["tokenizer_savedir"]
epoch = yaml_args["epoch"]
max_sequenge_length = yaml_args["max_sequenge_length"]


# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(modelname, pad_token="<|pad|>")

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.save_pretrained(tokenizer_savedir)


# %%
# get the dataset
import pandas as pd
import torch

# same as the bart we used 20 90% query
def trunction_query(row):
    return " ".join(row.split()[:input_length])


def trunction_target(row):
    return " ".join(row.split()[:target_length])


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
            f"{data_path}/{data_type}.source", sep="\t", names=["source"]
        )
        self.df_target = pd.read_csv(
            f"{data_path}/{data_type}.target", sep="\t", names=["target"]
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


# %%
train_dataset = S2Sdataset(max_length=max_sequenge_length, data_type="train")
valid_dataset = S2Sdataset(max_length=max_sequenge_length, data_type="val")
# test_dataset = S2Sdataset(max_length=200, data_type="test")

# class CustomTrainer(Trainer):
# def evaluate(
#     self, eval_dataset=valid_dataset, ignore_keys=[""], metric_key_prefix="eval"
# ):

#     source_list = []
#     target_list = []
#     for text in eval_dataset:
#         source_list.append(text["source"])
#         target_list.append(text["target"])

#     encode_texts = tokenizer(
#         source_list, truncation=True, max_length=input_length, padding="max_length"
#     )
#     sample_outputs = model.generate(
#         encode_texts.input_ids,
#         do_sample=False,
#         num_beams=1,
#         max_length=60,
#         pad_token_id=tokenizer.eos_token_id,
#     )

#     print(sample_outputs)

#     result = calculate_rouge(source_list, target_list)
#     logger.info("rouge1:" + str(result["rouge1"].mid.fmeasure))
#     return {"eval_rouge": result["rouge1"].mid.fmeasure}

# def evaluate(
#     self, eval_dataset=valid_dataset, ignore_keys=[""], metric_key_prefix="eval"
# ):
#     super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

#     target_list = []
#     source_list = []
#     for text in tqdm(eval_dataset):

#         prompt = text["source"]
#         target_list.append(text["target"])
#         # note that we only want to make the parameters same as the bart, so we set the max_length of query to 20.
#         generated = tokenizer(
#             f"{prompt}", return_tensors="pt", max_length=target_length
#         ).input_ids.cuda()

#         # perform prediction
#         sample_outputs = model.generate(
#             generated,
#             do_sample=False,
#             top_k=50,
#             max_length=60,
#             top_p=0.90,
#             temperature=0,
#             num_return_sequences=0,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#         # decode the predicted tokens into texts
#         predicted_text = tokenizer.decode(
#             sample_outputs[0], skip_special_tokens=True
#         )
#         predicted_text = predicted_text.strip().replace("\n", "")
#         expand_text = " ".join(
#             predicted_text.split("<|pad|>")[-1].split(" ")[:target_length]
#         )
#         source_list.append(expand_text)
#     result = calculate_rouge(source_list, target_list)
#     logger.info("rouge1:" + str(result["rouge1"].mid.fmeasure))
#     return {"eval_rouge": result["rouge1"].mid.fmeasure}


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epoch,
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

# %%
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
model.resize_token_embeddings(len(tokenizer))

# %%
# start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# trainer.train()

# %%
# Test


def test():
    fout = open(f"{predict_data_path}", "w")
    model = GPT2LMHeadModel.from_pretrained(yaml_args["save_model"]).cuda()
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


test()
