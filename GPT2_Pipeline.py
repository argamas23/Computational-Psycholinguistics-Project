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
    """
    Revised to handle sub-word tokenization and 1024-token limit 
    more gracefully to avoid the indexing errors seen in logs. 
    """
    surprisals = []
    
    # Pre-tokenize the whole sentence to ensure alignment
    full_text = " ".join(words_list)
    
    for i in range(len(words_list)):
        if i == 0:
            surprisals.append(None) # No context for the first word
            continue
            
        # Context is everything before the current word
        context = " ".join(words_list[:i])
        target = " " + words_list[i] # GPT-2 expects a leading space for mid-sentence words

        # Encode context and target
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)

        # Truncate context to fit within the 1024 limit 
        # We leave room for the target tokens
        max_context = 1024 - len(target_ids)
        if len(context_ids) > max_context:
            context_ids = context_ids[-max_context:]

        input_ids = torch.tensor([context_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits # Shape: [1, seq_len, vocab_size]

        # Get the logits for the very last token of the context to predict target
        last_token_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(last_token_logits, dim=0)

        # If word is split into multiple tokens, surprisal = sum of individual surprisals
        # based on the chain rule: P(w1, w2) = P(w1) * P(w2|w1)
        word_surprisal = 0.0
        temp_context_ids = list(context_ids)

        for t_id in target_ids:
            input_ids = torch.tensor([temp_context_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_ids)
                log_probs = torch.log_softmax(outputs.logits[0, -1, :], dim=0)
            
            word_surprisal += -log_probs[t_id].item()
            temp_context_ids.append(t_id) # Update context for multi-token words

        surprisals.append(word_surprisal)

    return surprisals

# def compute_surprisal_for_sentence(words_list):
#     surprisals = []
#     # Using a sliding window approach to avoid the 1024 truncation 
#     for i in range(len(words_list)):
#         if i == 0:
#             surprisals.append(np.nan)
#             continue
            
#         # Context window: Max 512 tokens to stay safe within GPT-2 limits
#         context = " ".join(words_list[max(0, i-50):i]) 
#         target = " " + words_list[i]
        
#         inputs = tokenizer(context, return_tensors="pt").to(device)
#         target_ids = tokenizer.encode(target, add_special_tokens=False)
        
#         word_surprisal = 0.0
#         curr_input_ids = inputs["input_ids"]
        
#         for t_id in target_ids:
#             with torch.no_grad():
#                 outputs = model(curr_input_ids)
#                 # Proper log_softmax to get surprisal [cite: 23]
#                 log_probs = torch.log_softmax(outputs.logits[0, -1, :], dim=0)
#                 word_surprisal += -log_probs[t_id].item()
#                 # Append for multi-token word chain rule: P(w1,w2|ctx) = P(w1|ctx)P(w2|ctx,w1)
#                 curr_input_ids = torch.cat([curr_input_ids, torch.tensor([[t_id]]).to(device)], dim=1)
        
#         surprisals.append(word_surprisal)
#     return surprisals

# FIX: Log-transform RT before Bayesian analysis for normality 
df_avg["log_RT"] = np.log(df_avg["RT"])

# =====================================
# STEP 4: APPLY TO DATA
# =====================================
df_avg["gpt2_surprisal"] = None

grouped = df_avg.groupby("item")

for item_id, group in tqdm(grouped):
    words_list = group["word"].astype(str).tolist()
    surprisals = compute_surprisal_for_sentence(words_list)
    df_avg.loc[group.index, "gpt2_surprisal"] = surprisals

print("GPT-2 surprisal computed")

# =====================================
# STEP 5: CLEAN
# =====================================
df_clean = df_avg.dropna().reset_index(drop=True)
df_clean["gpt2_surprisal"] = pd.to_numeric(df_clean["gpt2_surprisal"], errors="coerce")
df_clean = df_clean.dropna()

print("After cleaning:", df_clean.shape)

# =====================================
# STEP 6: SAVE
# =====================================
df_clean.to_csv("gpt2_surprisal.csv", index=False)

# =====================================
# STEP 7: CORRELATION
# =====================================
pearson_corr, p_val = pearsonr(df_clean["gpt2_surprisal"], df_clean["RT"])
spearman_corr, _ = spearmanr(df_clean["gpt2_surprisal"], df_clean["RT"])

print(f"Pearson: {pearson_corr:.4f} (p={p_val:.4f})")
print(f"Spearman: {spearman_corr:.4f}")

# =====================================
# STEP 8: PLOT
# =====================================
plt.figure(figsize=(10, 6))
plt.scatter(df_clean["gpt2_surprisal"], df_clean["RT"], alpha=0.3)
plt.xlabel("GPT-2 Surprisal")
plt.ylabel("Reading Time")
plt.title("GPT-2 Surprisal vs RT")
plt.show()

# =====================================
# STEP 9: BAYESIAN REGRESSION
# =====================================
data = df_clean.copy()

# Standardize features
X = (data["gpt2_surprisal"] - data["gpt2_surprisal"].mean()) / data["gpt2_surprisal"].std()
Y = (data["RT"] - data["RT"].mean()) / data["RT"].std()

with pm.Model() as bayes_model:
    beta = pm.Normal("beta", 0, 1)
    intercept = pm.Normal("intercept", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)

    mu = intercept + beta * X.values
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y.values)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9) 

print(az.summary(trace)) 
az.plot_posterior(trace)
plt.show()