import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model_pl import pl_model
from data_loader_pl import S2SDataModule
from model_plrl import ReinforcementModel
from pathlib import Path


datamodule = S2SDataModule()

# model = pl_model(datamodule.vocab)
model = ReinforcementModel(datamodule.vocab)

filename = "{epoch}-{val_rouge:.4f}"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # dirpath=args.output_dir / "save_model", filename="{epoch}-{val_loss:.2f}-{val_map:.2f}", monitor=args.ckpt_metric, mode=args.ckpt_mode,
    dirpath=Path("lstm_ppo") / "save_model",
    filename=filename,
    monitor="val_rouge",
    mode="max",
    save_top_k=1,
    every_n_epochs=None,
    # every_n_train_steps=100,
    # save_last=True,
)


trainer = Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback],
    max_epochs=150,
    # resume_from_checkpoint="lstm_lstm/save_model/epoch=97-val_rouge=0.08.ckpt",
)
# trainer.fit(
#     model, datamodule, ckpt_path="lstm_lstm/save_model/epoch=97-val_rouge=0.08.ckpt"
# )
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
    ckpt_path="lstm_ppo/save_model/epoch=104-val_rouge=0.08.ckpt",
)
