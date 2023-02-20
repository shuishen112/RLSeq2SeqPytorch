import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch as T
import torch.nn as nn
from data_util import data, config
import os
from model import Encoder, Decoder, Model
from beam_search import beam_search, beam_search_pl
from train_util import get_enc_data, get_cuda, get_dec_data
from rouge import Rouge
import yaml

yaml_args = yaml.load(open("yaml_config/nq_lstm_answer.yaml"), Loader=yaml.FullLoader)


class pl_model(pl.LightningModule):
    def __init__(self, vocab) -> None:
        super().__init__()

        # get the model
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
        # self.model = Model()
        self.vocab = vocab
        self.start_id = self.vocab.__getitem__(data.START_DECODING)
        self.end_id = self.vocab.__getitem__(data.STOP_DECODING)
        self.pad_id = self.vocab.__getitem__(data.PAD_TOKEN)
        self.unk_id = self.vocab.__getitem__(data.UNKNOWN_TOKEN)

        self.lr = 0.001
        self.print_sents = True
        self.save_hyperparameters()

    def forward(
        self,
        enc_batch,
        enc_lens,
        enc_padding_mask,
        ct_e,
        extra_zeros,
        enc_batch_extend_vocab,
        max_dec_len,
        dec_batch,
    ):

        enc_embed = self.embeds(enc_batch)  # Get embeddings for encoder input
        enc_out, enc_hidden = self.encoder(enc_embed, enc_lens)

        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = get_cuda(
            T.LongTensor(len(enc_out)).fill_(self.start_id)
        )  # Input to the decoder

        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        final_dist_list = []
        for t in range(min(max_dec_len, config.max_dec_steps)):
            use_gound_truth = get_cuda(
                (T.rand(len(enc_out)) > 0.25)
            ).long()  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            x_t = (
                use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t
            )  # Select decoder input based on use_ground_truth probabilities
            x_t = self.embeds(x_t)
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.decoder(
                x_t,
                s_t,
                enc_out,
                enc_padding_mask,
                ct_e,
                extra_zeros,
                enc_batch_extend_vocab,
                sum_temporal_srcs,
                prev_s,
            )

            final_dist_list.append(final_dist)

            x_t = T.multinomial(
                final_dist, 1
            ).squeeze()  # Sample words from final distribution which can be used as input in next time step
            is_oov = (
                x_t >= config.vocab_size
            ).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (
                is_oov
            ) * self.unk_id  # Replace OOVs with [UNK] token
        return final_dist_list

    def training_step(self, batch, batch_idx):

        (
            enc_batch,
            enc_lens,
            enc_padding_mask,
            enc_batch_extend_vocab,
            extra_zeros,
            context,
        ) = get_enc_data(batch)

        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)

        # enc_out, enc_hidden = self(enc_batch, enc_lens)
        final_dist_list = self(
            enc_batch,
            enc_lens,
            enc_padding_mask,
            context,
            extra_zeros,
            enc_batch_extend_vocab,
            max_dec_len,
            dec_batch,
        )

        # Get input and target batchs for training decoder
        step_losses = []

        for t in range(len(final_dist_list)):
            final_dist = final_dist_list[t]
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(
                log_probs, target, reduction="none", ignore_index=self.pad_id
            )
            step_losses.append(step_loss)

        losses = T.sum(
            T.stack(step_losses, 1), 1
        )  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  # Average batch loss

        self.log(
            "loss/train",
            mle_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=config.batch_size,
        )
        return mle_loss

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents):
        filename = "test_debug" + ".txt"

        with open(os.path.join(yaml_args["save_model_path"], filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def validation_step(self, batch, batch_idx):

        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()

        (
            enc_batch,
            enc_lens,
            enc_padding_mask,
            enc_batch_extend_vocab,
            extra_zeros,
            ct_e,
        ) = get_enc_data(batch)

        enc_batch = self.embeds(enc_batch)
        enc_out, enc_hidden = self.encoder(enc_batch, enc_lens)

        # -----------------------Summarization----------------------------------------------------

        pred_ids = beam_search_pl(
            enc_hidden,
            enc_out,
            enc_padding_mask,
            ct_e,
            extra_zeros,
            enc_batch_extend_vocab,
            self.embeds,
            self.decoder,
            self.start_id,
            self.end_id,
            self.unk_id,
        )

        for i in range(len(pred_ids)):
            decoded_words = data.outputids2words_new(
                pred_ids[i], self.vocab, batch.art_oovs[i]
            )
            if len(decoded_words) < 2:
                decoded_words = "xxx"
            else:
                decoded_words = " ".join(decoded_words)
            decoded_sents.append(decoded_words)
            abstract = batch.original_abstracts[i]
            article = batch.original_articles[i]
            ref_sents.append(abstract)
            article_sents.append(article)

        if self.print_sents:
            self.print_original_predicted(
                decoded_sents,
                ref_sents,
                article_sents,
            )

        scores = rouge.get_scores(decoded_sents, ref_sents, avg=True)
        rouge_l = torch.tensor(scores["rouge-l"]["f"])
        # print("rouge-l", scores["rouge-l"]["f"])
        return {"rouge-l": rouge_l}

        # self.log("rouge_l", rouge_l, on_step=False, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):

        decoded_sents = []
        ref_sents = []
        article_sents = []
        (
            enc_batch,
            enc_lens,
            enc_padding_mask,
            enc_batch_extend_vocab,
            extra_zeros,
            ct_e,
        ) = get_enc_data(batch)

        enc_batch = self.embeds(enc_batch)
        enc_out, enc_hidden = self.encoder(enc_batch, enc_lens)

        # -----------------------Summarization----------------------------------------------------

        pred_ids = beam_search_pl(
            enc_hidden,
            enc_out,
            enc_padding_mask,
            ct_e,
            extra_zeros,
            enc_batch_extend_vocab,
            self.embeds,
            self.decoder,
            self.start_id,
            self.end_id,
            self.unk_id,
        )

        for i in range(len(pred_ids)):
            decoded_words = data.outputids2words_new(
                pred_ids[i], self.vocab, batch.art_oovs[i]
            )
            if len(decoded_words) < 2:
                decoded_words = "xxx"
            else:
                decoded_words = " ".join(decoded_words)
            decoded_sents.append(decoded_words)
            abstract = batch.original_abstracts[i]
            article = batch.original_articles[i]
            ref_sents.append(abstract)
            article_sents.append(article)

        return {"generated_terms": decoded_sents}

    def test_epoch_end(self, outputs):
        # pred_l = []
        fout = open(f"{yaml_args['data_dir']}/test_generated_lstm.txt", "w")
        for output_batch in outputs:
            # pred_l.extend(output_batch["generated_terms"])
            for generated_terms in output_batch["generated_terms"]:
                fout.write(generated_terms + "\n")
                fout.flush()

    def validation_epoch_end(self, outputs):
        avg_rouge_l = torch.stack([x["rouge-l"] for x in outputs]).mean()
        print("avg_rouge_l", avg_rouge_l)
        self.log("val_rouge", avg_rouge_l, on_epoch=True)

    def configure_optimizers(self):
        # opt_g = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        # opt_d = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        # return [opt_g, opt_d], []

        return torch.optim.Adam(self.parameters(), lr=self.lr)
