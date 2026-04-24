# =====================================
# STEP 0: IMPORTS
# =====================================
import pandas as pd
import numpy as np
import math
import nltk
from nltk import PCFG, ViterbiParser, induce_pcfg
from nltk.corpus import treebank
from tqdm import tqdm
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =====================================
# STEP 1: LOAD & PREPROCESS DATA
# =====================================
# Matches the data loading strategy in your other pipelines
rt = pd.read_csv("../naturalstories/naturalstories_RTS/processed_RTs.tsv", sep="\t")
words = pd.read_csv("../naturalstories/naturalstories_RTS/processed_wordinfo.tsv", sep="\t")

df = pd.merge(words[["item", "zone", "word"]], rt[["item", "zone", "RT", "WorkerId"]], on=["item", "zone"])

# Average RTs across participants and Log-transform for normality
df_avg = df.groupby(["item", "zone", "word"])["RT"].mean().reset_index()
df_avg = df_avg[df_avg["RT"].between(100, 2000)]
df_avg["log_RT"] = np.log(df_avg["RT"]) # Log-RT for psycholinguistic scaling
df_avg = df_avg.sort_values(by=["item", "zone"]).reset_index(drop=True)

# =====================================
# STEP 2: INDUCE PCFG
# =====================================
nltk.download('treebank')
productions = []
for tree in treebank.parsed_sents():
    tree.collapse_unary(collapsePOS=True, collapseRoot=True)
    tree.chomsky_normal_form()
    productions += tree.productions()

grammar = induce_pcfg(nltk.Nonterminal('S'), productions)
parser = ViterbiParser(grammar)

# =====================================
# STEP 3: CORRECTED SURPRISAL FUNCTION
# =====================================
def get_pcfg_surprisal(sentence_tokens):
    surprisals = []
    for i in range(len(sentence_tokens)):
        prefix = [t.lower() for t in sentence_tokens[:i+1]]
        try:
            parses = list(parser.parse(prefix))
            if parses:
                # FIX: Using -log(P) to ensure Surprisal Theory alignment [cite: 23, 52]
                prob = max(parses[0].prob(), 1e-12) 
                surprisals.append(-math.log(prob))
            else:
                surprisals.append(None)
        except:
            surprisals.append(None)
    return surprisals

# =====================================
# STEP 4: APPLY AND DEFINE DF_CLEAN
# =====================================
df_avg["pcfg_surprisal"] = None
grouped = df_avg.groupby("item")

for item_id, group in tqdm(grouped, desc="Computing PCFG"):
    words_list = group["word"].astype(str).tolist()
    df_avg.loc[group.index, "pcfg_surprisal"] = get_pcfg_surprisal(words_list)

# FIX: Define df_clean BEFORE standardizing
df_clean = df_avg.dropna(subset=["pcfg_surprisal"]).copy()
df_clean["pcfg_surprisal"] = pd.to_numeric(df_clean["pcfg_surprisal"])

# Standardize features for Bayesian regression stability [cite: 35]
df_clean["pcfg_std"] = (df_clean["pcfg_surprisal"] - df_clean["pcfg_surprisal"].mean()) / df_clean["pcfg_surprisal"].std()
df_clean["log_rt_std"] = (df_clean["log_RT"] - df_clean["log_RT"].mean()) / df_clean["log_RT"].std()

# =====================================
# STEP 5: STATS & PLOTTING
# =====================================
p_corr, p_val = pearsonr(df_clean["pcfg_surprisal"], df_clean["log_RT"])
print(f"Corrected PCFG Pearson: {p_corr:.4f} (p={p_val:.4f})")

# Using distinct colors as per your preference for clarity
plt.figure(figsize=(8, 5))
plt.scatter(df_clean["pcfg_surprisal"], df_clean["log_RT"], alpha=0.4, color='teal', label='PCFG Data')
plt.title("Syntactic Surprisal vs Log Reading Time")
plt.xlabel("PCFG Surprisal (-log P)")
plt.ylabel("Log RT")
plt.legend()
plt.savefig("pcfg_correlation.png")

# =====================================
# STEP 6: BAYESIAN REGRESSION
# =====================================
with pm.Model() as pcfg_model:
    beta = pm.Normal("beta", 0, 1)
    intercept = pm.Normal("intercept", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    
    mu = intercept + beta * df_clean["pcfg_std"].values
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=df_clean["log_rt_std"].values)
    
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

print(az.summary(trace))
df_clean.to_csv("pcfg_surprisal_final.csv", index=False)