# %%
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer

# %%
# load dataset
import pandas as pd

# %%
# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

modelname = "gpt2"

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    modelname,
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
)

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.save_pretrained("./models/tokenizer/")


# %%
# get the dataset
import pandas as pd
import torch


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

        df_merge = (
            "<|startoftext|>Source:"
            + df_source["source"]
            + "<|pad|>Target:"
            + df_target["target"]
            + "<|endoftext|>"
        )
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
    load_best_model_at_end=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="logs",
)

# %%
model = GPT2LMHeadModel.from_pretrained("results/checkpoint-3700").cuda()
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
# ).valid()

# %%
# Test

fout = open("scifact/gpt_predict.txt", "w")


def test():
    model.eval()

    df = pd.read_csv("scifact/test.source", names=["source"], sep="\t")
    for text in df["source"].to_list():
        prompt = f"<|startoftext|>Source: {text}\nTarget:"
        generated = tokenizer(f"{prompt}", return_tensors="pt").input_ids.cuda()

        # perform prediction
        sample_outputs = model.generate(
            generated,
            do_sample=False,
            top_k=50,
            max_length=70,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        predicted_text = predicted_text.strip().replace("\n", "")
        target = predicted_text.split("Target:")[-1]

        fout.write(target + "\n")
        fout.flush()
        # print("predicted_text", predicted_text)


# test()
