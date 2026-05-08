import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.stats import pearsonr
import pymc as pm
import arviz as az

# ==========================================================
# STEP 1: CONSOLIDATE DATA (FIXING THE INNER JOIN BOTTLENECK)
# ==========================================================
def consolidate_data_fixed():
    print("Consolidating datasets using Left Join...")
    # Load your pipeline outputs
    df_tri = pd.read_csv("trigram_surprisal_final.csv")
    df_pcfg = pd.read_csv("pcfg_surprisal_final.csv")
    df_gpt2 = pd.read_csv("gpt2_surprisal.csv")

    # Use GPT-2 as the anchor to avoid the 'inner join' bottleneck
    # This preserves all 10,246 data points from your neural model
    df = df_gpt2[['item', 'zone', 'word', 'log_RT', 'gpt2_surprisal']].merge(
        df_tri[['item', 'zone', 'surprisal']], on=['item', 'zone'], how='left'
    ).rename(columns={'surprisal': 'trigram_surprisal'})
    
    df = df.merge(
        df_pcfg[['item', 'zone', 'pcfg_surprisal']], on=['item', 'zone'], how='left'
    )

    # Locality Info: Word position within the story
    df['locality_pos'] = df.groupby('item').cumcount()

    # Standardize all columns for the Bayesian model
    cols = ['gpt2_surprisal', 'trigram_surprisal', 'pcfg_surprisal', 'locality_pos', 'log_RT']
    for col in cols:
        df[f'{col}_std'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

# ==========================================================
# STEP 2: COMPARATIVE VISUALIZATION (3 DISTINCT SEGMENTS)
# ==========================================================
def plot_final_comparison(df):
    # Colors chosen for high contrast between segments: 
    # Lexical (Red), Structural (Teal), Neural (Green)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # 1. Lexical (Trigram)
    valid_tri = df.dropna(subset=['trigram_surprisal_std'])
    sns.regplot(data=valid_tri, x='trigram_surprisal_std', y='log_RT_std', 
                ax=axes[0], color='#FF7F7F', scatter_kws={'alpha': 0.1})
    r_tri, _ = pearsonr(valid_tri['trigram_surprisal_std'], valid_tri['log_RT_std'])
    axes[0].set_title(f"Lexical: Trigram\\nPearson r = {r_tri:.3f}")

    # 2. Structural (PCFG)
    valid_pcfg = df.dropna(subset=['pcfg_surprisal_std'])
    sns.regplot(data=valid_pcfg, x='pcfg_surprisal_std', y='log_RT_std', 
                ax=axes[1], color='#008080', scatter_kws={'alpha': 0.1})
    r_pcfg, _ = pearsonr(valid_pcfg['pcfg_surprisal_std'], valid_pcfg['log_RT_std'])
    axes[1].set_title(f"Structural: PCFG\\nPearson r = {r_pcfg:.3f}")

    # 3. Neural (GPT-2)
    valid_gpt = df.dropna(subset=['gpt2_surprisal_std'])
    sns.regplot(data=valid_gpt, x='gpt2_surprisal_std', y='log_RT_std', 
                ax=axes[2], color='#2E8B57', scatter_kws={'alpha': 0.1})
    r_gpt, _ = pearsonr(valid_gpt['gpt2_surprisal_std'], valid_gpt['log_RT_std'])
    axes[2].set_title(f"Neural: GPT-2\\nPearson r = {r_gpt:.3f}")

    plt.suptitle("Final Comparison: Surprisal Theory vs Reading Time", fontsize=16)
    plt.savefig("final_analysis_fixed.png")

# ==========================================================
# STEP 3: BAYESIAN REGRESSION (SURPRISAL + LOCALITY)
# ==========================================================
def run_bayesian_locality(df):
    print("Running Bayesian Analysis...")
    # Removing NaNs specifically for the regression variables
    data = df.dropna(subset=['gpt2_surprisal_std', 'locality_pos_std', 'log_RT_std'])

    with pm.Model() as final_model:
        # Priors for Surprisal and Locality
        beta_surp = pm.Normal("beta_surp", mu=0, sigma=1)
        beta_loc = pm.Normal("beta_loc", mu=0, sigma=1)
        alpha = pm.Normal("intercept", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", 1)
        
        # Linear Predictor: log_RT ~ Surprisal + Locality
        mu = alpha + beta_surp * data['gpt2_surprisal_std'] + beta_loc * data['locality_pos_std']
        
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data['log_RT_std'])
        
        # Sampling using NUTS
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    print(az.summary(trace, hdi_prob=0.95))
    az.plot_forest(trace, var_names=["beta_surp", "beta_loc"], combined=True)
    plt.savefig("final_bayesian_forest.png")

# ==========================================================
# STEP 4: ERROR ANALYSIS (IDENTIFYING HARD SENTENCES)
# ==========================================================
def identify_difficult_sentences(df):
    # Calculate Residuals (Actual - Predicted by Surprisal)
    # We use the mean beta of ~0.25 found in your previous logs
    df['error'] = np.abs(df['log_RT_std'] - (0.248 * df['gpt2_surprisal_std']))
    
    # Sort and pick top 10 most difficult words to predict
    difficult_words = df.sort_values(by='error', ascending=False).head(10)
    print("--- Error Analysis: Top 10 Difficult Words ---")
    print(difficult_words[['word', 'gpt2_surprisal', 'log_RT', 'error']])
    difficult_words.to_csv("difficult_sentences_analysis.csv", index=False)

if __name__ == "__main__":
    final_df = consolidate_data_fixed()
    plot_final_comparison(final_df)
    run_bayesian_locality(final_df)
    identify_difficult_sentences(final_df)
    print("Analysis complete. Saved: final_analysis_fixed.png and final_bayesian_forest.png")