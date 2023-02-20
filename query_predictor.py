from rouge_score import rouge_scorer, scoring
import pandas as pd
import pickle
import math
import numpy as np
from gensim.utils import tokenize as gensim_tokenize
import pyterrier as pt

pt.init()
# idf = pickle.load(open("notebook/idf.pickle", "rb"))
# word_count = pickle.load(open("notebook/word_count.pickle", "rb"))

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def calculate_rouge(output_lns, reference_lns):

    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return result


def get_rouge_f1_score(output_lns, reference_lns):

    return [
        scorer.score(reference_ln, output_ln)["rouge1"].fmeasure
        for reference_ln, output_ln in zip(reference_lns, output_lns)
    ]


def IDF(t):
    if t in idf.vocabulary_:
        return idf.idf_[idf.vocabulary_[t]]
    else:
        return 0


def SCG(t, term_frequency, idf_cal):
    return (1 + math.log(term_frequency[t] + 1)) * idf_cal(t)


# print(SCG("of", word_count, IDF))


def MaxSCQ(query):
    seg_list = [SCG(token, word_count, IDF) for token in query.split()]
    if len(seg_list) == 0:
        return 0.0
    else:
        return np.max(seg_list)


def get_maxscq(querys):
    l = [MaxSCQ(query) for query in querys]
    return l


def AvgSCG(query):
    seg_list = [SCG(token, word_count, IDF) for token in query.split()]
    if len(seg_list) == 0:
        return 0.0
    else:
        return np.mean(seg_list)


def SumSCQ(query):
    seg_list = [SCG(token, word_count, IDF) for token in query.split()]
    if len(seg_list) == 0:
        return 0.0
    else:
        return np.sum(seg_list)


# 对query进行预处理
def clean(row):
    text = row["query"].strip().lower()
    tokens = list(gensim_tokenize(text))
    text = " ".join(tokens)
    return text


def get_retrieval_material():

    # 检索
    index = pt.IndexFactory.of(
        "/projects/futhark1/data/wzm289/code/GAR/gar/indices/beir_scifact"
    )
    print(index.getCollectionStatistics().toString())
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    train_dataset = pt.get_dataset("irds:beir/scifact/train")
    train_qrel = train_dataset.get_qrels()
    querys = train_dataset.get_topics("text")

    test_dataset = pt.get_dataset("irds:beir/scifact/test")
    test_qrel = test_dataset.get_qrels()
    # print(querys)
    # print(querys.dtypes)
    # result = pt.Experiment(
    #     [bm25],
    #     querys,
    #     train_qrel,
    #     eval_metrics=["recall_100"],
    #     verbose=True,
    # )
    # print(result)

    return bm25, train_qrel, train_qrel, test_qrel


class QueryReward:
    def __init__(self, reward_name, reward_type="pre-retrieval"):
        self.reward_name = reward_name
        self.reward_type = reward_type
        if self.reward_type == "post-retrieval":
            (
                self.bm25,
                self.train_qrel,
                self.val_qrel,
                self.test_qrel,
            ) = get_retrieval_material()

    def get_metric_reward(self, qids, preds, data_type, reward_name="ndcg_cut_10"):
        if data_type == "train":
            qrel_data = self.train_qrel
        elif data_type == "val":
            qrel_data = self.val_qrel
        elif data_type == "test":
            qrel_data = self.test_qrel

        df_result = pd.DataFrame({"qid": qids, "query": preds})
        df_result = df_result.astype({"qid": object})

        df_result["query"] = df_result.apply(clean, axis=1)

        def isnan(x):
            if x == x:
                return False
            else:
                return True

        eval_result = pt.Experiment(
            [self.bm25],
            df_result,
            qrel_data,
            eval_metrics=[reward_name],
            perquery=True,
        )

        reward_scores = []

        for q in qids:
            if q in eval_result["qid"].to_list():
                score = eval_result.loc[eval_result.qid == q, "value"].to_list()[0]
                if isnan(score):
                    reward_scores.append(0.0)
                else:
                    reward_scores.append(score)
            else:
                print("not in evaluation?")
                reward_scores.append(0.0)

        return reward_scores

    def get_reward_score(
        self, generated_query, target_query, qids=None, source_text=None, data_type=None
    ):
        if self.reward_type == "pre-retrieval":
            if self.reward_name == "val-ROUGE-1":
                return get_rouge_f1_score(generated_query, target_query)
            elif self.reward_name == "MaxSCQ":
                return get_maxscq(generated_query)
        elif self.reward_type == "post-retrieval":
            return self.get_metric_reward(
                qids, generated_query, data_type, self.reward_name
            )


# df = pd.read_csv("datasets/scifact/test.csv", sep="\t").head()

# qw = QueryReward("MaxSCQ")
# print(qw.get_reward_score(df["query"].to_list(), df["text"].to_list()))
# print(get_rouge_f1_score(df["query"].to_list(), df["text"].to_list()))
