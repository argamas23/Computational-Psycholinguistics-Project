# =====================================
# STEP 0: IMPORTS
# =====================================
import pandas as pd
import numpy as np
import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =====================================
# STEP 1: LOAD DATA (SAME AS YOUR CODE)
# =====================================
rt = pd.read_csv("../naturalstories/naturalstories_RTS/processed_RTs.tsv", sep="\t")
words = pd.read_csv("../naturalstories/naturalstories_RTS/processed_wordinfo.tsv", sep="\t")

words_clean = words[["item", "zone", "word"]]
rt_clean = rt[["item", "zone", "RT", "WorkerId"]]

df = pd.merge(words_clean, rt_clean, on=["item", "zone"])

df_avg = df.groupby(["item", "zone", "word"])["RT"].mean().reset_index()
df_avg = df_avg[df_avg["RT"] > 0]
df_avg = df_avg.sort_values(by=["item", "zone"]).reset_index(drop=True)

print("Dataset ready:", df_avg.shape)

# =====================================
# STEP 2: LOAD GPT-2
# =====================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("GPT-2 loaded on:", device)

# =====================================
# STEP 3: FUNCTION TO COMPUTE SURPRISAL
# =====================================
def compute_surprisal_for_sentence(words_list):
    surprisals = []

    for i in range(1, len(words_list)):
        context = " ".join(words_list[:i])
        target = words_list[i]

        # Tokenize FULL context
        inputs = tokenizer(context, return_tensors="pt")
        
        # 🔥 FIX: truncate to last 1024 tokens
        if inputs["input_ids"].shape[1] > 1024:
            inputs["input_ids"] = inputs["input_ids"][:, -1024:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -1024:]

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits[0, -1], dim=0)

        target_ids = tokenizer.encode(target, add_special_tokens=False)

        if len(target_ids) == 1:
            prob = probs[target_ids[0]].item()
            surprisal = -math.log(prob)
        else:
            surprisal = None

        surprisals.append(surprisal)

    surprisals.insert(0, None)
    return surprisals

# =====================================
# STEP 4: APPLY TO DATA
# =====================================
df_avg["gpt2_surprisal"] = None

grouped = df_avg.groupby("item")

for item_id, group in tqdm(grouped):
    words_list = group["word"].tolist()
    surprisals = compute_surprisal_for_sentence(words_list)

    df_avg.loc[group.index, "gpt2_surprisal"] = surprisals

print("GPT-2 surprisal computed")

# =====================================
# STEP 5: CLEAN
# =====================================
df_clean = df_avg.dropna().reset_index(drop=True)

print("After cleaning:", df_clean.shape)

print(df_clean.head())

# =====================================
# STEP 6: SAVE
# =====================================
df_clean.to_csv("gpt2_surprisal.csv", index=False)
df_clean["gpt2_surprisal"] = pd.to_numeric(df_clean["gpt2_surprisal"], errors="coerce")
df_clean = df_clean.dropna()

# =====================================
# STEP 7: CORRELATION
# =====================================
pearson_corr, p_val = pearsonr(df_clean["gpt2_surprisal"], df_clean["RT"])
spearman_corr, _ = spearmanr(df_clean["gpt2_surprisal"], df_clean["RT"])

print("Pearson:", pearson_corr)
print("Spearman:", spearman_corr)

# =====================================
# STEP 8: PLOT
# =====================================
plt.figure()
plt.scatter(df_clean["gpt2_surprisal"], df_clean["RT"], alpha=0.3)
plt.xlabel("GPT-2 Surprisal")
plt.ylabel("Reading Time")
plt.title("GPT-2 Surprisal vs RT")
plt.show()

# =====================================
# STEP 9: BAYESIAN REGRESSION
# =====================================
data = df_clean.copy()

X = (data["gpt2_surprisal"] - data["gpt2_surprisal"].mean()) / data["gpt2_surprisal"].std()
Y = (data["RT"] - data["RT"].mean()) / data["RT"].std()

X = X.values
Y = Y.values

with pm.Model() as model:

    beta = pm.Normal("beta", 0, 1)
    intercept = pm.Normal("intercept", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)

    mu = intercept + beta * X

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

print(az.summary(trace))

az.plot_posterior(trace)