"""
Generates a 1x3 subplot figure for Gradient Variance vs. Patch Radius.
Aggregates data automatically from all patch_variance_results.jsonl files.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

def generate_3panel_plot():
    print("[*] Scanning for JSONL results...")
    
    data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    file_list = glob.glob("results/**/patch_variance_results.jsonl", recursive=True)
    
    if not file_list:
        print("[!] No 'patch_variance_results.jsonl' files found.")
        return

    for file_path in file_list:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    ds_type = data.get("dataset_type", "").lower()
                    init_val = data.get("init", "").lower()
                    n = data.get("n")
                    r = data.get("radius_r")
                    var = data.get("var")
                    
                    # Filter for your exact datasets
                    if ds_type not in ["product_bernoulli", "binary_mixture", "ising"]: continue
                    if n not in [9, 16, 25]: continue
                    
                    if "identity" in init_val in init_val:
                        init_name = "Identity"
                    elif "data" in init_val:
                        init_name = "Data-Dependent"
                    else:
                        continue 
                        
                    if r is not None and var is not None:
                        data_store[ds_type][init_name][n][r].append(var)
                except json.JSONDecodeError:
                    pass

    # Aggregate Means
    plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ds in data_store:
        for init in data_store[ds]:
            for n in data_store[ds][init]:
                radii = sorted(data_store[ds][init][n].keys())
                for r in radii:
                    mean_var = np.mean(data_store[ds][init][n][r])
                    plot_data[ds][init][n].append((r, mean_var))

    # Setup Plot
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm"
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # Exact Dataset Titles
    datasets_to_plot = [
        ('product_bernoulli', 'Product Bernoulli (Random)'), 
        ('binary_mixture', 'Blob Dataset (Structured)'), 
        ('ising', 'Ising Model (Structured)')
    ]
    
    colors = {'Identity': "#ff8e05dd", 'Data-Dependent': '#1f77b4'}
    markers = {9: 'o', 16: '^', 25: 's'}
    
    for i, (ds_key, ds_title) in enumerate(datasets_to_plot):
        ax = axes[i]
        
        if ds_key not in plot_data:
            ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(ds_title, fontsize=16)
            continue
            
        for init_name in ['Identity', 'Data-Dependent']:
            if init_name not in plot_data[ds_key]: continue
            
            for n in [9, 16, 25]:
                if n not in plot_data[ds_key][init_name]: continue
                
                xy_data = plot_data[ds_key][init_name][n]
                x_vals = [item[0] for item in xy_data]
                y_vals = [item[1] for item in xy_data]
                
                ax.plot(
                    x_vals, y_vals, 
                    marker=markers[n], markersize=8,
                    markeredgecolor='black', markeredgewidth=1.0,
                    linestyle='-', linewidth=2, 
                    color=colors[init_name], alpha=0.85
                )

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(ds_title, fontsize=16, pad=10)
        ax.set_xlabel(r'Patch Radius ($r$)', fontsize=14)
        
        if i == 0:
            ax.set_ylabel(r'$\mathrm{Var}_\theta[\partial_j \mathcal{L}(\theta)]$', fontsize=14)
            
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which="major", ls="-", alpha=0.15, color='gray')
        ax.grid(True, which="minor", ls=":", alpha=0.08, color='gray')
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Global Custom Legend
    legend_elements = [
        Line2D([0], [0], color=colors['Identity'], lw=2.5, label='Identity'),
        Line2D([0], [0], color=colors['Data-Dependent'], lw=2.5, label='Data-Dependent'),
        Line2D([0], [0], color='gray', marker='o', markeredgecolor='black', linestyle='', markersize=9, label='$n = 9$'),
        Line2D([0], [0], color='gray', marker='^', markeredgecolor='black', linestyle='', markersize=9, label='$n = 16$'),
        Line2D([0], [0], color='gray', marker='s', markeredgecolor='black', linestyle='', markersize=9, label='$n = 25$')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), 
               ncol=5, fontsize=14, frameon=True, edgecolor='black', framealpha=1.0)
    
    plt.tight_layout()
    out_path = "results/combined_patch.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    print(f"[+] Multi-dataset plot successfully saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    generate_3panel_plot()