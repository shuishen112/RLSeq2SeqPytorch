# %%
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
# %%
# load dataset
import pandas as pd

# %%
# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

modelname = "gpt2"

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(modelname, pad_token="<|pad|>")

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.save_pretrained("./models/tokenizer/")


# %%
# get the dataset
import pandas as pd
import torch

# same as the bart we used 20 90% query
def trunction_query(row):
    return " ".join(row["source"].split()[:20])


class S2Sdataset(Dataset):
    def __init__(self, max_length=20, data_type="train"):
        self.input_ids = []
        self.attn_masks = []

        df_source = pd.read_csv(
            f"scifact/{data_type}.source", sep="\t", names=["source"]
        )
        df_target = pd.read_csv(
            f"scifact/{data_type}.target", sep="\t", names=["target"]
        )
        df_source["source"] = df_source.apply(trunction_query, axis=1)
        df_merge = df_source["source"] + "<|pad|>" + df_target["target"]

        for item in df_merge:
            # tokenize
            encodings_dict = tokenizer(
                item, truncation=True, max_length=max_length, padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attn_masks[index]


# %%
train_dataset = S2Sdataset(max_length=40, data_type="train")
valid_dataset = S2Sdataset(max_length=40, data_type="val")
# test_dataset = S2Sdataset(max_length=200, data_type="test")


training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=100,
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
)

# %%
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
model.resize_token_embeddings(len(tokenizer))

# %%
# start training
# Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     data_collator=lambda data: {
#         "input_ids": torch.stack([f[0] for f in data]),
#         "attention_mask": torch.stack([f[1] for f in data]),
#         "labels": torch.stack([f[0] for f in data]),
#     },
# ).train()

# %%
# Test


fout = open("scifact/gpt_predict.txt", "w")




def test():
    model = GPT2LMHeadModel.from_pretrained("results/checkpoint-1200").cuda()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    df = pd.read_csv("scifact/test.source", names=["source"], sep="\t")
    for text in tqdm(df["source"].to_list()):
        prompt = text

        # note that we only want to make the parameters same as the bart, so we set the max_length of query to 20.

        generated = tokenizer(
            f"{prompt}", return_tensors="pt", max_length=20
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
        target = " ".join(predicted_text.split("<|pad|>")[-1].split(" ")[:12])
        fout.write(target + "\n")
        fout.flush()
        # print("predicted_text", predicted_text)


test()
