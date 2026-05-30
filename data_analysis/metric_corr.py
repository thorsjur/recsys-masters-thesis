import pandas as pd

path = "output/results/experiments.jsonl"

df = pd.read_json(path, lines=True)

test = pd.json_normalize(df["test_results"])

corr = test.corr(method="spearman", numeric_only=True)["ndcg@5"].drop("ndcg@5").sort_values(ascending=False)

print(corr)
