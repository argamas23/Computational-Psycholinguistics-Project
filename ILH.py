import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

# ==========================================================
# STEP 1: LOAD & PREPARE DATA WITH LOCALITY
# ==========================================================
def load_ilh_data():
    # Load your generated results
    df_tri = pd.read_csv("trigram_surprisal_final.csv")
    df_pcfg = pd.read_csv("pcfg_surprisal_final.csv")
    df_gpt2 = pd.read_csv("gpt2_surprisal.csv")

    # Anchor to GPT-2 to fix the bottleneck issue (Left Join)
    df = df_gpt2[['item', 'zone', 'word', 'log_RT', 'gpt2_surprisal']].merge(
        df_tri[['item', 'zone', 'surprisal']], on=['item', 'zone'], how='left'
    ).rename(columns={'surprisal': 'trigram_surprisal'})
    
    df = df.merge(
        df_pcfg[['item', 'zone', 'pcfg_surprisal']], on=['item', 'zone'], how='left'
    )

    # Define Locality: Distance from start of sentence/story (Integration Hint)
    df['locality_hint'] = df.groupby('item').cumcount()

    # Standardize for Bayesian comparison [cite: 35]
    for col in ['gpt2_surprisal', 'trigram_surprisal', 'pcfg_surprisal', 'locality_hint', 'log_RT']:
        df[f'{col}_std'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

# ==========================================================
# STEP 2: MULTI-MODEL ILH BAYESIAN REGRESSION
# ==========================================================
def run_ilh_comparison(df):
    results = {}
    
    # Model Configurations with your requested color segments:
    # Lexical (Red), Structural (Teal), Neural (Green)
    model_configs = [
        ('trigram_surprisal_std', 'Lexical_ILH', '#FF7F7F'),
        ('pcfg_surprisal_std', 'Structural_ILH', '#008080'),
        ('gpt2_surprisal_std', 'Neural_ILH', '#2E8B57')
    ]

    for surp_col, name, color in model_configs:
        print(f"Running {name}...")
        # Drop NaNs for the specific model to maximize data points
        data = df.dropna(subset=[surp_col, 'locality_hint_std', 'log_RT_std'])
        
        with pm.Model() as ilh_model:
            # Priors [cite: 35, 36]
            b_surp = pm.Normal("beta_surp", 0, 1)
            b_loc = pm.Normal("beta_loc", 0, 1)
            intercept = pm.Normal("intercept", 0, 1)
            sigma = pm.HalfNormal("sigma", 1)
            
            # ILH Linear Model: RT ~ Surprisal + Locality
            mu = intercept + b_surp * data[surp_col].values + b_loc * data['locality_hint_std'].values
            
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data['log_RT_std'].values)
            
            trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
            results[name] = trace

    return results

# ==========================================================
# STEP 3: VISUALIZE ILH IMPACT
# ==========================================================
def plot_ilh_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Custom colors for segments: Lexical, Structural, Neural
    colors = ['#FF7F7F', '#008080', '#2E8B57']
    
    for i, (name, trace) in enumerate(results.items()):
        # Plot posterior distributions for beta coefficients
        az.plot_forest(trace, var_names=["beta_surp", "beta_loc"], 
                       ax=axes[i], combined=True, colors=colors[i])
        axes[i].set_title(f"ILH Analysis: {name}")
        axes[i].axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.savefig("ilh_comparison_forest.png")

if __name__ == "__main__":
    df_master = load_ilh_data()
    ilh_traces = run_ilh_comparison(df_master)
    plot_ilh_results(ilh_traces)
    print("ILH Analysis complete. Check 'ilh_comparison_forest.png' for results.")