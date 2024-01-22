from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers.utils import logging
from gpt2_training.config import config
from gpt2_training.dataset_loader import S2Sdataset
from torch.utils.data import Dataset, DataLoader
import torch
from rouge_score import rouge_scorer, scoring
import pandas as pd
import os

os.environ["WANDB_DISABLED"] = "true"


def calculate_rouge(output_lns, reference_lns):
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return result


logging.set_verbosity_info()
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
logger = logging.get_logger("transformers")

# test the gpt tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.modelname, pad_token="<|pad|>")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = S2Sdataset(args, tokenizer=tokenizer, data_type="train")
    valid_dataset = S2Sdataset(args, tokenizer=tokenizer, data_type="val")

    class CustomTrainer(Trainer):
        def evaluate(
            self, eval_dataset=valid_dataset, ignore_keys=[""], metric_key_prefix="eval"
        ):
            valid_dataloader = DataLoader(eval_dataset, batch_size=30, shuffle=False)

            all_generated_terms = []
            target_list = []
            for batch in valid_dataloader:
                target_list.extend(batch["target"])
                inputs = tokenizer(
                    batch["source"],
                    truncation=True,
                    max_length=args.target_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                sample_outputs = model.generate(
                    input_ids=inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                    do_sample=False,
                    max_length=60,
                    pad_token_id=tokenizer.eos_token_id,
                )

                generated_tokens = tokenizer.batch_decode(
                    sample_outputs, skip_special_tokens=True
                )

                generated_terms = []
                for item in generated_tokens:
                    predicted_text = item.strip().replace("\n", "")
                    print("predicted_text:", predicted_text)
                    expand_text = " ".join(
                        predicted_text.split("<|sep|>")[-1].split(" ")[
                            : args.target_length
                        ]
                    )
                    generated_terms.append(expand_text)

                all_generated_terms.extend(generated_terms)

            result = calculate_rouge(all_generated_terms, target_list)

            logger.info("rouge1:" + str(result["rouge1"].mid.fmeasure))
            return {"eval_rouge": result["rouge1"].mid.fmeasure}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epoch,
        logging_steps=args.logging_steps,
        load_best_model_at_end=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        # weight_decay=0.01,
        logging_dir="logs",
        save_total_limit=1,  # only save one file
        learning_rate=1e-3
        # report_to="none",
        # metric_for_best_model="rouge",
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    model.resize_token_embeddings(len(tokenizer))

    trainer = CustomTrainer(
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
    qid = 0
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
            max_length=args.max_sequence_length,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        predicted_text = predicted_text.strip().replace("\n", "")
        print(predicted_text)
        target = " ".join(
            predicted_text.split("<|sep|>")[-1].split(" ")[: args.target_length]
        )

        # concate to the original text to form the new query
        new_query = target
        fout.write(str(qid) + "\t" + new_query + "\n")
        qid += 1
        fout.flush()


if __name__ == "__main__":
    if config.train:
        train(args=config)
    elif config.predict:
        predict(args=config)
