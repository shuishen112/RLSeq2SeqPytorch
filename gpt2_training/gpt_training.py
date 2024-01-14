from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers.utils import logging
from gpt2_training.config import config
from gpt2_training.dataset_loader import S2Sdataset


import torch

import pandas as pd


logging.set_verbosity_info()
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
logger = logging.get_logger("transformers")

# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.modelname, pad_token="<|pad|>")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = S2Sdataset(args, tokenizer=tokenizer, data_type="train")
    valid_dataset = S2Sdataset(args, tokenizer=tokenizer, data_type="val")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epoch,
        # logging_steps=args.logging_steps,
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


def predict(args):
    fout = open(f"{args.predict_data_path}", "w")
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(args.modelname, pad_token="<|pad|>")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    df = pd.read_csv(args.inference_input, names=["source"], sep="\t")
    for text in tqdm(df["source"].to_list()):
        prompt = text

        # note that we only want to make the parameters same as the bart, so we set the max_length of query to 20.
        generated = tokenizer(
            f"{prompt}", return_tensors="pt", max_length=args.input_length
        ).input_ids.to(device)

        # perform prediction
        sample_outputs = model.generate(
            generated,
            do_sample=False,
            top_k=50,
            max_length=args.max_sequenge_length,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        predicted_text = predicted_text.strip().replace("\n", "")
        target = " ".join(
            predicted_text.split("<|pad|>")[-1].split(" ")[: args.target_length]
        )
        fout.write(target + "\n")
        fout.flush()
        # print("predicted_text", predicted_text)


if __name__ == "__main__":
    if config.train:
        train(args=config)
    elif config.predict:
        predict(args=config)
