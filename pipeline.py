# =====================================
# STEP 0: IMPORTS
# =====================================
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import pymc as pm
import arviz as az


# =====================================
# STEP 1: LOAD DATA
# =====================================
rt = pd.read_csv("../naturalstories/naturalstories_RTS/processed_RTs.tsv", sep="\t")
words = pd.read_csv("../naturalstories/naturalstories_RTS/processed_wordinfo.tsv", sep="\t")

print("Loaded datasets")

# =====================================
# STEP 2: CLEAN + SELECT REQUIRED COLUMNS
# =====================================
words_clean = words[["item", "zone", "word"]]
rt_clean = rt[["item", "zone", "RT", "WorkerId"]]

# =====================================
# STEP 3: MERGE DATA
# =====================================
df = pd.merge(words_clean, rt_clean, on=["item", "zone"])

print("Merged dataset shape:", df.shape)

# =====================================
# STEP 4: AVERAGE READING TIMES
# =====================================
df_avg = df.groupby(["item", "zone", "word"])["RT"].mean().reset_index()

# Remove invalid RTs
df_avg = df_avg[df_avg["RT"] > 0]

# Sort properly
df_avg = df_avg.sort_values(by=["item", "zone"]).reset_index(drop=True)

print("After averaging:", df_avg.shape)

# =====================================
# STEP 5: ADD CONTEXT
# =====================================
df_avg["prev_word"] = df_avg["word"].shift(1)
df_avg["prev2_word"] = df_avg["word"].shift(2)

# Reset context at story boundaries
df_avg.loc[df_avg["item"] != df_avg["item"].shift(1), ["prev_word", "prev2_word"]] = None

# Drop rows without full context
df_clean = df_avg.dropna().reset_index(drop=True)

print("After adding context:", df_clean.shape)

# =====================================
# STEP 6: BUILD VOCAB + HANDLE UNKNOWN WORDS
# =====================================
# Optional: replace rare words with UNK
word_counts = df_clean["word"].value_counts()

# threshold (you can tune this later)
UNK_THRESHOLD = 2

vocab = set(word_counts[word_counts > UNK_THRESHOLD].index)

def replace_unk(word):
    return word if word in vocab else "<UNK>"

df_clean["word"] = df_clean["word"].apply(replace_unk)
df_clean["prev_word"] = df_clean["prev_word"].apply(replace_unk)
df_clean["prev2_word"] = df_clean["prev2_word"].apply(replace_unk)

# Final vocab size
V = len(set(df_clean["word"]))
print("Vocabulary size:", V)

# =====================================
# STEP 7: BUILD TRIGRAM MODEL
# =====================================
trigram_counts = defaultdict(Counter)
bigram_counts = Counter()

for _, row in df_clean.iterrows():
    w = row["word"]
    w1 = row["prev_word"]
    w2 = row["prev2_word"]

    trigram_counts[(w2, w1)][w] += 1
    bigram_counts[(w2, w1)] += 1

print("Trigram model built")

# =====================================
# STEP 8: SMOOTHED PROBABILITY
# =====================================
def trigram_prob(w, w1, w2):
    # Add-one (Laplace) smoothing
    numerator = trigram_counts[(w2, w1)][w] + 1
    denominator = bigram_counts[(w2, w1)] + V
    return numerator / denominator

# =====================================
# STEP 9: SURPRISAL COMPUTATION
# =====================================
def surprisal(w, w1, w2):
    prob = trigram_prob(w, w1, w2)
    return -math.log(prob)

# Compute surprisal
df_clean["surprisal"] = df_clean.apply(
    lambda row: surprisal(row["word"], row["prev_word"], row["prev2_word"]),
    axis=1
)

print("Surprisal computed")

# =====================================
# STEP 10: FINAL CHECKS
# =====================================
print(df_clean.head())

print("\nSurprisal stats:")
print(df_clean["surprisal"].describe())

# =====================================
# STEP 11: SAVE OUTPUT
# =====================================
df_clean.to_csv("clean_with_surprisal.csv", index=False)

print("\nSaved: clean_with_surprisal.csv")

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Remove any NaNs (just in case)
df_analysis = df_clean.dropna()

# ==============================
# Correlation
# ==============================
pearson_corr, p_val = pearsonr(df_analysis["surprisal"], df_analysis["RT"])
spearman_corr, _ = spearmanr(df_analysis["surprisal"], df_analysis["RT"])

print("Pearson correlation:", pearson_corr)
print("p-value:", p_val)
print("Spearman correlation:", spearman_corr)

# ==============================
# Scatter Plot
# ==============================
plt.figure()
plt.scatter(df_analysis["surprisal"], df_analysis["RT"], alpha=0.3)

plt.xlabel("Surprisal")
plt.ylabel("Reading Time (RT)")
plt.title("Surprisal vs Reading Time")

plt.show()

# Prepare data
data = df_clean.dropna()

# Normalize (VERY IMPORTANT for stability)
surprisal = (data["surprisal"] - data["surprisal"].mean()) / data["surprisal"].std()
rt = (data["RT"] - data["RT"].mean()) / data["RT"].std()

# Convert to numpy
X = surprisal.values
Y = rt.values

# ==============================
# Bayesian Linear Regression
# ==============================

with pm.Model() as model:

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    sigma = pm.HalfNormal("sigma", 1)

    # Linear model
    mu = intercept + beta * X

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

    # Sampling
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# ==============================
# RESULTS
# ==============================

print(az.summary(trace, hdi_prob=0.95))

# Plot posterior
az.plot_posterior(trace)