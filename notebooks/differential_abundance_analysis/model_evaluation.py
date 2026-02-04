import numpy as np
import pandas as pd

import scanpy as sc

import scipy as sp
from sklearn.metrics import precision_recall_curve, auc

def compute_precision_and_recall_auc(predicted, true):
    # Compute precision and recall curve and AUC
    precision, recall, _ = precision_recall_curve(true, predicted)
    pr_auc = auc(recall, precision)
    return pr_auc

def cluster_metrics(abs_ratios, ratios, is_abundant, leiden_labels):
    # Non-abundant clusters should always be lower 
    non_abundant_over_abundant = abs_ratios[is_abundant].mean() / abs_ratios[~is_abundant].mean()
    # Compute fraction correct sign
    ratios_abundant = ratios[is_abundant]  # Only clusters 1 and 2 
    leiden_abundant = np.array(leiden_labels)[is_abundant]

    leiden_abundant_unique = np.unique(leiden_abundant)
    assert "0" not in leiden_abundant_unique
    assert "3" not in leiden_abundant_unique
    
    ratios_abundant[leiden_abundant=="2"] = ratios_abundant[leiden_abundant=="2"] * (-1)
    correct_sign_prop = (ratios_abundant >= 0).sum() / len(ratios_abundant)
    
    # Kolmogorov-Smirnov test 
    ks_D, _ = sp.stats.ks_2samp(
        abs_ratios[is_abundant],
        abs_ratios[~is_abundant],
        alternative="two-sided",
        mode="asymp"
    )
    w_distance = sp.stats.wasserstein_distance(abs_ratios[is_abundant], abs_ratios[~is_abundant])
    return non_abundant_over_abundant, correct_sign_prop, ks_D, w_distance
    
def compute_evaluation_metrics(result_path, model, subsampling_rates, n_runs):
    results_per_oversamp = {
        "p_major_cluster": [], 
        "p_minor_cluster": [],
        "auc_score": [], 
        "non_abundant_over_abundant": [],
        "correct_sign_prop": [],
        "ks_D": [], 
        "w_distance": [],
        "run": []
        }
    
    results_per_run = {
        "metric": [], 
        "value": [],
        "run": []
        }

    for i in range(n_runs):
        # Compute the ratios and abundances per cluster
        per_cluster_ratios = []
        per_cluster_log_odds = []
        
        for r_oversamp in subsampling_rates:
            # Compute the true per-cluster ratio 
            p = 0.5 - r_oversamp  # Always minor cluster prop
            one_minus_p = 1 - p  # Always major cluster prop
    
            # Collect results and compute AUC 
            if model=="mrvi":
                adata_generated = sc.read_h5ad(result_path / f"oversamp_{r_oversamp}_{i}" / f"oversamp_{r_oversamp}_{i}.h5ad")
            elif model=="meld":
                adata_generated = sc.read_h5ad(result_path / f"oversamp_{r_oversamp}.h5ad")
            else:
                adata_generated = sc.read_h5ad(result_path / f"oversamp_{r_oversamp}_{i}.h5ad")
            
            # Collect abundance labels 
            is_abundant = np.logical_or(adata_generated.obs.leiden=="1", adata_generated.obs.leiden=="2")
            if model=="mrvi":
                abs_ratio = np.abs(adata_generated.obs[f"log_ratio_oversamp_{r_oversamp}_{i}"]).values
                ratio = adata_generated.obs[f"log_ratio_oversamp_{r_oversamp}_{i}"].values
            elif model=="scRatio":
                abs_ratio = np.abs(adata_generated.obs["log_ratios"]).values
                ratio = adata_generated.obs["log_ratios"].values
            else:
                abs_ratio = np.abs(adata_generated.obs["log_ratio"]).values
                ratio = adata_generated.obs["log_ratio"].values

            if model=="milo":
                adata_generated = adata_generated[~np.isnan(abs_ratio)]
                is_abundant = is_abundant[~np.isnan(abs_ratio)]
                abs_ratio = abs_ratio[~np.isnan(abs_ratio)]
                ratio = ratio[~np.isnan(ratio)]

            # Compute AUC
            auc = compute_precision_and_recall_auc(abs_ratio, is_abundant)
            results_per_oversamp["p_major_cluster"].append(one_minus_p)
            results_per_oversamp["p_minor_cluster"].append(p)
            results_per_oversamp["auc_score"].append(auc)
            results_per_oversamp["run"].append(i)
            
            for cl in ["0", "1", "2", "3"]:
                per_cluster_ratios.append(np.mean(ratio[adata_generated.obs["leiden"]==cl]))

                numerator = (adata_generated.obs.loc[adata_generated.obs["leiden"]==cl, "treatment"]==1).sum()
                denominator = (adata_generated.obs.loc[adata_generated.obs["leiden"]==cl, "treatment"]==0).sum() 

                abs_log_odds = np.log((numerator + 1e-9) / (denominator + 1e-9))
                per_cluster_log_odds.append(abs_log_odds)
            
            # Compute cluster metrics
            abundant_over_non_abundant, correct_sign_prop, ks_D, w_distance = cluster_metrics(abs_ratio, ratio, is_abundant, adata_generated.obs["leiden"])
            results_per_oversamp["non_abundant_over_abundant"].append(abundant_over_non_abundant)
            results_per_oversamp["correct_sign_prop"].append(correct_sign_prop)
            results_per_oversamp["ks_D"].append(ks_D)
            results_per_oversamp["w_distance"].append(w_distance)
        
        results_per_run["metric"].append("spearman_log_odds")
        results_per_run["run"].append(i)
        results_per_run["value"].append(sp.stats.spearmanr(per_cluster_ratios, per_cluster_log_odds)[0])
        results_per_run["metric"].append("pearson_log_odds")
        results_per_run["run"].append(i)
        results_per_run["value"].append(sp.stats.pearsonr(per_cluster_ratios, per_cluster_log_odds)[0])

    # After computing the metrics per run and abundance, compute metrics per run only. 
    results_per_oversamp_df = pd.DataFrame(results_per_oversamp)
    for i in range(n_runs):
        # Take df at run i 
        results_per_oversamp_df_i = results_per_oversamp_df.loc[results_per_oversamp_df.run==i]
        for metric in results_per_oversamp_df_i.columns:
            name = f"corr_{metric}_p"
            results_per_run["metric"].append(name)
            results_per_run["run"].append(i) 
            results_per_run["value"].append(sp.stats.spearmanr(results_per_oversamp_df_i[metric],
                                                 results_per_oversamp_df_i["p_major_cluster"])[0])

            name = f"mean_{metric}"
            results_per_oversamp_df_over_i_03 = results_per_oversamp_df_i[results_per_oversamp_df_i.p_major_cluster>0.8]
            results_per_run["metric"].append(name)
            results_per_run["run"].append(i) 
            results_per_run["value"].append(results_per_oversamp_df_over_i_03[metric].mean())
        
    # Results per data frame 
    results_per_run_df = pd.DataFrame(results_per_run)    
    return results_per_oversamp_df, results_per_run_df
