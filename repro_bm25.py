import urllib.request
import json
import tqdm
import pytrec_eval

from typing import Dict, Tuple
from datasets import load_dataset
from pyserini.search import SimpleSearcher
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--split", type=str, default="trec_dl2019")
argparser.add_argument(
    "--use_expand", action="store_true", help="use expand query or not"
)
argparser.add_argument(
    "--expand_query_path",
    type=str,
    default="data/TREC/trec-2019/gpt_predict.tsv",
    help="the path of the expand query",
)

args = argparser.parse_args()

expand_querys = open(f"{args.expand_query_path}", "r").readlines()


def trec_eval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int] = (10, 50, 100, 200, 1000),
) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def load_qrels_from_url(url: str) -> Dict[str, Dict[str, int]]:
    qrels = {}
    for line in urllib.request.urlopen(url).readlines():
        qid, _, pid, score = line.decode("utf-8").strip().split()
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][pid] = int(score)

    print(
        "Load {} queries {} qrels from {}".format(
            len(qrels), sum(len(v) for v in qrels.values()), url
        )
    )
    return qrels


def main(split: str = "trec_dl2019"):
    searcher: SimpleSearcher = SimpleSearcher.from_prebuilt_index("msmarco-passage")

    query2doc_dataset = load_dataset("intfloat/query2doc_msmarco")[split]

    # fout_source = open("data/TREC/trec-2020/test.source", "w")
    # fout_target = open("data/TREC/trec-2020/test.target", "w")

    queries = []
    for idx in range(len(query2doc_dataset)):
        example = query2doc_dataset[idx]

        if args.use_expand:
            e_query = expand_querys[idx].strip().split("\t")[-1]
            print(e_query)
            new_query = "{} {}".format(
                " ".join([example["query"] for _ in range(5)]), e_query
            )
        else:
            new_query = "{} {}".format(
                " ".join([example["query"] for _ in range(5)]), example["pseudo_doc"]
            )
        queries.append(example["query"])
        # fout_source.write(example["query"] + "\n")
        # fout_target.write(example["pseudo_doc"] + "\n")
    print("Load {} queries".format(len(queries)))

    results: Dict[str, Dict[str, float]] = {}
    batch_size = 64
    num_batches = (len(queries) + batch_size - 1) // batch_size
    for i in tqdm.tqdm(range(num_batches), mininterval=2):
        batch_query_ids = query2doc_dataset["query_id"][
            i * batch_size : (i + 1) * batch_size
        ]
        batch_queries = queries[i * batch_size : (i + 1) * batch_size]
        qid_to_hits: dict = searcher.batch_search(
            batch_queries, qids=batch_query_ids, k=1000, threads=8
        )
        for qid, hits in qid_to_hits.items():
            results[qid] = {hit.docid: hit.score for hit in hits}

    split_to_qrels_url = {
        "trec_dl2019": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        "trec_dl2020": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        "validation": "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv",
    }
    qrels = load_qrels_from_url(split_to_qrels_url[split])
    all_metrics = trec_eval(qrels=qrels, results=results)

    print("Evaluation results for {} split:".format(split))
    print(json.dumps(all_metrics, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main(split=args.split)
