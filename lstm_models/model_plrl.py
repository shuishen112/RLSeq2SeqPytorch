import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch as T
import torch.nn as nn
from torch.distributions import Categorical
from data_util import data, config
import os
from model import Encoder, Decoder, Model
from beam_search import beam_search, beam_search_pl
from train_util import get_enc_data, get_cuda, get_dec_data
from rouge import Rouge


class ReinforcementModel(pl.LightningModule):
    def __init__(self, vocab) -> None:
        super().__init__()

        # get the model
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)

        self.vocab = vocab
        self.start_id = self.vocab.__getitem__(data.START_DECODING)
        self.end_id = self.vocab.__getitem__(data.STOP_DECODING)
        self.pad_id = self.vocab.__getitem__(data.PAD_TOKEN)
        self.unk_id = self.vocab.__getitem__(data.UNKNOWN_TOKEN)

        self.lr = 0.001
        self.print_sents = True
        self.train_rl = True
        self.train_mle = True
        self.mle_weight = 0.25
        self.save_hyperparameters()

    def train_batch_MLE(
        self,
        enc_out,
        enc_hidden,
        enc_padding_mask,
        ct_e,
        extra_zeros,
        enc_batch_extend_vocab,
        batch,
    ):
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)

        # Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = get_cuda(
            T.LongTensor(len(enc_out)).fill_(self.start_id)
        )  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
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
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(
                log_probs, target, reduction="none", ignore_index=self.pad_id
            )
            step_losses.append(step_loss)
            x_t = T.multinomial(
                final_dist, 1
            ).squeeze()  # Sample words from final distribution which can be used as input in next time step

            is_oov = (
                x_t >= config.vocab_size
            ).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (
                is_oov
            ) * self.unk_id  # Replace OOVs with [UNK] token

        losses = T.sum(
            T.stack(step_losses, 1), 1
        )  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  # Average batch loss
        return mle_loss

    def train_batch_RL(
        self,
        enc_out,
        enc_hidden,
        enc_padding_mask,
        ct_e,
        extra_zeros,
        enc_batch_extend_vocab,
        article_oovs,
        greedy,
    ):
        s_t = enc_hidden

        x_t = get_cuda(
            T.LongTensor(len(enc_out)).fill_(self.start_id)
        )  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []  # Stores sampled indices for each time step
        decoder_padding_mask = []  # Stores padding masks of generated samples
        log_probs = []  # Stores log probabilites of generated samples
        mask = get_cuda(
            T.LongTensor(len(enc_out)).fill_(1)
        )  # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

        for t in range(config.max_dec_steps):
            x_t = self.embeds(x_t)  # [200,256] note that the embedding size is 256
            probs, s_t, ct_e, sum_temporal_srcs, prev_s = self.decoder(
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
            # probs: [batch, vocab_size]
            # s_t [batch]

            # for each step, it will sample a log_prob
            if greedy is False:
                multi_dist = Categorical(probs)
                x_t = multi_dist.sample()  # perform multinomial sampling
                log_prob = multi_dist.log_prob(x_t)
                log_probs.append(log_prob)
            else:
                _, x_t = T.max(probs, dim=1)  # perform greedy sampling
            x_t = x_t.detach()
            inds.append(x_t)
            mask_t = get_cuda(
                T.zeros(len(enc_out))
            )  # Padding mask of batch for current time step
            mask_t[
                mask == 1
            ] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[
                (mask == 1) + (x_t == self.end_id) == 2
            ] = 0  # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (
                x_t >= config.vocab_size
            ).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t + (
                is_oov
            ) * self.unk_id  # Replace OOVs with [UNK] token

        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if (
            greedy is False
        ):  # If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1)
            log_probs = (
                log_probs * decoder_padding_mask
            )  # Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask, dim=1)  # Length of sampled sentence
            log_probs = (
                T.sum(log_probs, dim=1) / lens
            )  # (bs,)                                     #compute normalizied log probability of a sentence
        decoded_strs = []
        for i in range(len(enc_out)):
            id_list = inds[i].cpu().numpy()
            oovs = article_oovs[i]
            S = data.outputids2words_new(
                id_list, self.vocab, oovs
            )  # Generate sentence corresponding to sampled words
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if (
                len(S) < 2
            ):  # If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs

    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            print("Rouge failed for multi sentence evaluation.. Finding exact pair")
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                except Exception:
                    print("Error occured at:")
                    print("decoded_sents:", decoded_sents[i])
                    print("original_sents:", original_sents[i])
                    score = [{"rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
        return rouge_l_f1

    def training_step(self, batch, batch_idx):

        (
            enc_batch,
            enc_lens,
            enc_padding_mask,
            enc_batch_extend_vocab,
            extra_zeros,
            context,
        ) = get_enc_data(batch)

        enc_batch = self.embeds(enc_batch)  # Get embeddings for encoder input
        enc_out, enc_hidden = self.encoder(enc_batch, enc_lens)

        if self.train_mle is True:

            mle_loss = self.train_batch_MLE(
                enc_out,
                enc_hidden,
                enc_padding_mask,
                context,
                extra_zeros,
                enc_batch_extend_vocab,
                batch,
            )
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))

        # --------------RL training-----------------------------------------------------
        if self.train_rl is True:  # perform reinforcement learning training
            # multinomial sampling
            sample_sents, RL_log_probs = self.train_batch_RL(
                enc_out,
                enc_hidden,
                enc_padding_mask,
                context,
                extra_zeros,
                enc_batch_extend_vocab,
                batch.art_oovs,
                greedy=False,
            )
            with T.autograd.no_grad():
                # greedy sampling
                greedy_sents, _ = self.train_batch_RL(
                    enc_out,
                    enc_hidden,
                    enc_padding_mask,
                    context,
                    extra_zeros,
                    enc_batch_extend_vocab,
                    batch.art_oovs,
                    greedy=True,
                )

            sample_reward = self.reward_function(sample_sents, batch.original_abstracts)
            baseline_reward = self.reward_function(
                greedy_sents, batch.original_abstracts
            )
            # if iter%200 == 0:
            #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
            rl_loss = (
                -(sample_reward - baseline_reward) * RL_log_probs
            )  # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            rl_loss = T.mean(rl_loss)

            batch_reward = T.mean(sample_reward).item()
        else:
            rl_loss = get_cuda(T.FloatTensor([0]))
            batch_reward = 0

        # ------------------------------------------------------------------------------------
        self.loss = (
            self.mle_weight * mle_loss + rl_loss + (1 - self.mle_weight) * rl_loss
        )
        return self.loss

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents):
        filename = "test_debug" + ".txt"

        with open(os.path.join("scifact", filename), "w") as f:
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

        # self.log("rouge_l", rouge_l, on_step=False, prog_bar=True, logger=True)
        print("rouge_l:", "%.4f" % rouge_l)
        return {"rouge-l": rouge_l}

    def validation_epoch_end(self, outputs):
        avg_rouge_l = torch.stack([x["rouge-l"] for x in outputs]).mean()
        print(avg_rouge_l)
        self.log("val_rouge", avg_rouge_l, on_epoch=True)

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
        fout = open("scifact/test_generated_ppo.txt", "w")
        for output_batch in outputs:
            # pred_l.extend(output_batch["generated_terms"])
            for generated_terms in output_batch["generated_terms"]:
                fout.write(generated_terms + "\n")
                fout.flush()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
