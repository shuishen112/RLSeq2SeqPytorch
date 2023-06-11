import pytorch_lightning as pl
from pytorch_lightning import Trainer

from lstm_models.model_pl import pl_model
from lstm_models.data_loader_pl import S2SDataModule

# from lstm_models.model_plrl import ReinforcementModel

from lstm_models.model_plrl_post_reward import ReinforcementPostModel
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

import yaml


yaml_args = yaml.load(open("yaml_config/msmarco_lstm_rl.yaml"), Loader=yaml.FullLoader)
# wandb_logger = WandbLogger(
#     project="CIKM2023",
#     name="lstm-scifact",
#     # config=args,
# )

datamodule = S2SDataModule(
    data_dir=yaml_args["data_dir"],
    train_type=yaml_args["train_type"],
    yaml_args=yaml_args,
)

# model = pl_model(datamodule.vocab, yaml_args)
model = ReinforcementPostModel(datamodule.vocab,yaml_args)

filename = yaml_args["filename"]
ckpt_path = yaml_args["ckpt_path"]
save_model_path = yaml_args["save_model_path"]
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # dirpath=args.output_dir / "save_model", filename="{epoch}-{val_loss:.2f}-{val_map:.2f}", monitor=args.ckpt_metric, mode=args.ckpt_mode,
    dirpath=Path(save_model_path) / "save_model",
    filename=filename,
    monitor=yaml_args["monitor"],
    mode=yaml_args["mode"],
    save_top_k=1,
    every_n_epochs=None,
    every_n_train_steps=10,
    # save_last=True,
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback],
    max_epochs=yaml_args["max_epochs"],
    resume_from_checkpoint=ckpt_path,
    logger=None,
    # fast_dev_run=True,
    limit_train_batches=1,
    # limit_val_batches=1,

)

trainer.fit(model, datamodule, ckpt_path=ckpt_path)
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
    ckpt_path=ckpt_path,
)
