# =====================================
# STEP 0: IMPORTS
# =====================================
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =====================================
# STEP 1: LOAD DATA
# =====================================
# Loading the Natural Stories Corpus datasets [cite: 45, 47, 78]
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
# STEP 4: PREPROCESS & AVERAGE
# =====================================
# Normalize words (lowercase & strip punctuation) to ensure better trigram matching [cite: 79]
df["word"] = df["word"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Average RTs across participants [cite: 28, 70]
df_avg = df.groupby(["item", "zone", "word"])["RT"].mean().reset_index()

# Filter RT outliers (Standard psycholinguistic range: 100ms to 2000ms)
df_avg = df_avg[df_avg["RT"].between(100, 2000)]

# Log-transform RT for normality
df_avg["log_RT"] = np.log(df_avg["RT"])

# Sort properly by story item and zone [cite: 69]
df_avg = df_avg.sort_values(by=["item", "zone"]).reset_index(drop=True)
print("After averaging and filtering:", df_avg.shape)

# =====================================
# STEP 5: ADD CONTEXT
# =====================================
df_avg["prev_word"] = df_avg["word"].shift(1)
df_avg["prev2_word"] = df_avg["word"].shift(2)

# Reset context at story boundaries to prevent cross-story trigrams
df_avg.loc[df_avg["item"] != df_avg["item"].shift(1), ["prev_word", "prev2_word"]] = None

# Drop rows without full context (the first two words of every story)
df_clean = df_avg.dropna().reset_index(drop=True)

# =====================================
# STEP 6: BUILD VOCAB + HANDLE UNKNOWN WORDS
# =====================================
word_counts = df_clean["word"].value_counts()
UNK_THRESHOLD = 2 
vocab = set(word_counts[word_counts > UNK_THRESHOLD].index)

def replace_unk(word):
    return word if word in vocab else "<UNK>"

df_clean["word"] = df_clean["word"].apply(replace_unk)
df_clean["prev_word"] = df_clean["prev_word"].apply(replace_unk)
df_clean["prev2_word"] = df_clean["prev2_word"].apply(replace_unk)

V = len(set(df_clean["word"]))
print("Vocabulary size:", V)

# =====================================
# STEP 7: BUILD TRIGRAM MODEL
# =====================================
trigram_counts = defaultdict(Counter)
bigram_counts = Counter()

for _, row in df_clean.iterrows():
    w, w1, w2 = row["word"], row["prev_word"], row["prev2_word"]
    trigram_counts[(w2, w1)][w] += 1
    bigram_counts[(w2, w1)] += 1

print("Trigram model built")

# =====================================
# STEP 8: SMOOTHED PROBABILITY & SURPRISAL
# =====================================
def get_surprisal(w, w1, w2):
    # Laplace (Add-one) smoothing to handle zero-frequency trigrams [cite: 18, 23]
    prob = (trigram_counts[(w2, w1)][w] + 1) / (bigram_counts[(w2, w1)] + V)
    return -math.log(prob)

df_clean["surprisal"] = df_clean.apply(
    lambda row: get_surprisal(row["word"], row["prev_word"], row["prev2_word"]), axis=1
)

# =====================================
# STEP 9: CORRELATION ANALYSIS
# =====================================
pearson_corr, p_val = pearsonr(df_clean["surprisal"], df_clean["log_RT"])
spearman_corr, _ = spearmanr(df_clean["surprisal"], df_clean["log_RT"])

print(f"Pearson correlation: {pearson_corr:.4f} (p={p_val:.4e})")
print(f"Spearman correlation: {spearman_corr:.4f}")

# Plotting the relationship [cite: 30, 32]
plt.figure(figsize=(10, 6))
plt.scatter(df_clean["surprisal"], df_clean["log_RT"], alpha=0.3)
plt.xlabel("Trigram Surprisal")
plt.ylabel("Log Reading Time")
plt.title("Trigram Surprisal vs Log RT")
plt.show()

# =====================================
# STEP 10: BAYESIAN REGRESSION
# =====================================
# Standardize features for MCMC stability [cite: 35, 76]
X = (df_clean["surprisal"] - df_clean["surprisal"].mean()) / df_clean["surprisal"].std()
Y = (df_clean["log_RT"] - df_clean["log_RT"].mean()) / df_clean["log_RT"].std()

with pm.Model() as bayesian_model:
    # Priors [cite: 36, 38]
    beta = pm.Normal("beta", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", 1)
    
    # Linear model: mu = alpha + beta * surprisal
    mu = intercept + beta * X.values
    
    # Likelihood [cite: 35, 77]
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y.values)
    
    # Sampling using NUTS [cite: 70, 75]
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9)

# Results Summary [cite: 76, 88]
print(az.summary(trace, hdi_prob=0.95))
az.plot_posterior(trace)
plt.show()

# =====================================
# STEP 11: SAVE OUTPUT
# =====================================
df_clean.to_csv("trigram_surprisal_final.csv", index=False)
print("Saved: trigram_surprisal_final.csv")