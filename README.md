# Sequence to Sequence using Reinforcement Learning
Combining [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/pdf/1705.04304.pdf) and [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)

> The code is based: https://github.com/rohithreddy024/Text-Summarizer-Pytorch. 

We fixed some issues from the original version and add the pytorchlightning version. 

## Model Description
* LSTM based Sequence-to-Sequence model for Abstractive Summarization
* Pointer mechanism for handling Out of Vocabulary (OOV) words [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)
* Intra-temporal and Intra-decoder attention for handling repeated words [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)
* Self-critic policy gradient training along with MLE training [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)

## Data
* Download train and valid pairs (article, title) of OpenNMT provided Gigaword dataset from [here](https://github.com/harvardnlp/sent-summary)
* Copy files ```train.article.txt```, ```train.title.txt```, ```valid.article.filter.txt```and ```valid.title.filter.txt``` to ```data/unfinished``` folder
* Files are already preprcessed

The previous version use some tricks, so we have to convert the dataformat to .bin files. 

### [original version] Creating ```.bin``` files and vocab file
* The model accepts data in the form of ```.bin``` files.
* To convert ```.txt``` file into ```.bin``` file and chunk them further, run (requires Python 2 & Tensorflow):
```
python make_data_files.py
```
* You will find the data in ```data/chunked``` folder and vocab file in ```data``` folder

### [original version] Training
* As suggested in [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf), first pretrain the seq-to-seq model using MLE (with Python 3):
```
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0
```
* Next, find the best saved model on validation data by running (with Python 3):
```
python eval.py --task=validate --start_from=0005000.tar
```
* After finding the best model (lets say ```0100000.tar```) with high rouge-l f score, load it and run (with Python 3):
```
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=0100000.tar --new_lr=0.0001
```
for MLE + RL training (or)
```
python train.py --train_mle=no --train_rl=yes --mle_weight=0.0 --load_model=0100000.tar --new_lr=0.0001
```
for RL training

### Validation
* To perform validation of RL training, run (with Python 3):
```
python eval.py --task=validate --start_from=0100000.tar
```
### Testing
* After finding the best model of RL training (lets say ```0025000.tar```), evaluate it on test data & get all rouge metrics by running (with Python 3):
```
python eval.py --task=test --load_model=0025000.tar
```

### （Our）Results

We only report the score in our training environment. 
* Rouge scores obtained by using best MLE trained model on test set:  

|  | r | p | f | 
| :-----| ----: | :----: |:----: |
| rouge-1 | 0.4197 | 0.4816 | 0.4384
| rouge-2 | 0.2214 | 0.2535 | 0.2306
| rouge-l | 0.4007 | 0.4596 | 0.4186


## Pytorch lighting Version


## References
* [pytorch implementation of "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/atulkum/pointer_summarizer)
* [https://github.com/rohithreddy024/Text-Summarizer-Pytorch](https://github.com/rohithreddy024/Text-Summarizer-Pytorch)