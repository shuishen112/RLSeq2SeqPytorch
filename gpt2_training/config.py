import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    help="the path of the data",
    type=str,
    default="scifact",
)
parser.add_argument(
    "--modelname",
    help="the name of the model",
    type=str,
    default="gpt2",
)
parser.add_argument(
    "--predict_data_path",
    help="the path of the predict data",
    type=str,
    default="scifact/gpt_predict.txt",
)

parser.add_argument(
    "--checkpoint_path",
    help="the path of the checkpoint",
    type=str,
    default=None,
)

parser.add_argument(
    "--output_dir",
    help="the path of the output",
    type=str,
    default="results",
)

parser.add_argument(
    "--input_length",
    help="the length of the input",
    type=int,
    default=20,
)

parser.add_argument(
    "--target_length",
    help="the length of the target",
    type=int,
    default=40,
)

parser.add_argument(
    "--predict_length",
    help="the length of the predict",
    type=int,
    default=20,
)

parser.add_argument(
    "--inference_input",
    help="the inference input",
    type=str,
    default="scifact/test.source",
)

parser.add_argument(
    "--epoch",
    help="the epoch of the training",
    type=int,
    default=100,
)

parser.add_argument(
    "--max_sequence_length",
    help="the max length of the sequence",
    type=int,
    default=40,
)


parser.add_argument(
    "--logging_steps",
    help="the logging steps of the training",
    type=int,
    default=100,
)

parser.add_argument(
    "--train",
    help="train the model",
    action="store_true",
)

parser.add_argument(
    "--predict",
    help="predict the model",
    action="store_true",
)


config = parser.parse_args()
