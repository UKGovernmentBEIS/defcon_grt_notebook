import pandas as pd


df = pd.read_csv("harmbench_behaviors_text_all.csv")

for value in df["SemanticCategory"].unique(): # for each category of evaluation
    if value == "copyright":
        continue
    df_subset = df.loc[(df["FunctionalCategory"] == "standard") & (df["SemanticCategory"] == value), :]
    df_subset.to_csv(f"harmbench_behaviors_text_{value}.csv")