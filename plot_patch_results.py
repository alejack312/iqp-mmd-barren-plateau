import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_variance(jsonl_path: str, target_n: int):
    # 1. Load the data from your JSONL file
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    # Filter for the specific number of qubits 
    df_n = df[df['n'] == target_n]
    
    if df_n.empty:
        print(f"No data found for n={target_n}")
        return

    plt.figure(figsize=(10, 6))
    
    for init_scheme, group in df_n.groupby('init'):
        group = group.sort_values('radius_r')
        
        plt.plot(
            group['radius_r'], 
            group['var'], 
            marker='o', 
            linewidth=2, 
            label=init_scheme
        )

    # Format the axes 
    plt.xscale('log', base=2)
    plt.yscale('log', base=10) 
    
    plt.xlabel('Patch Radius (r)', fontsize=12)
    plt.ylabel('Gradient Variance', fontsize=12)
    plt.title(f'Patch Variance Scaling (n={target_n}, "Ising" Dataset)', fontsize=14)
    
    plt.legend(title='Initialization Scheme', fontsize=11)
    
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    output_filename = f'combined_patch_variance_n{target_n}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Saved combined plot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    jsonl_file = "results/patch_test/patch_variance_results.jsonl" 
    
    plot_combined_variance(jsonl_file, target_n=16)