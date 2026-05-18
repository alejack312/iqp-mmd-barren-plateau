import json
import matplotlib.pyplot as plt

def plot_variance_vs_radius(files_to_plot):
    print("Gathering variance data...")
    plt.figure(figsize=(10, 6))

    for label, file_path in files_to_plot.items():
        radii = []
        variances = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Extract radius and variance 
                    if 'radius_r' in data and 'var' in data:
                        radii.append(data['radius_r'])
                        variances.append(data['var'])
            
            if not radii:
                print(f"-> Warning: No data found in {file_path}")
                continue
                
            # Sort by radius 
            sorted_data = sorted(zip(radii, variances))
            radii = [x[0] for x in sorted_data]
            variances = [x[1] for x in sorted_data]

            plt.plot(radii, variances, marker='o', linewidth=2.5, label=label)
            print(f"-> Successfully loaded: {label}")
            
        except FileNotFoundError:
            print(f"-> Skipping '{label}': File not found at {file_path}")

    # Format the graph
    plt.xscale('log', base=2)  
    plt.yscale('log', base=10)  
    
    plt.title("Barren Plateau Analysis: Gradient Variance vs. Patch Radius", fontsize=14, fontweight='bold')
    plt.xlabel("Patch Radius (r)", fontsize=12)
    plt.ylabel("Variance (Log Scale)", fontsize=12)
    plt.legend(title="Initialization Method", fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    # Save the graph
    output_name = "variance_vs_radius_proof.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Graph saved as '{output_name}'")


if __name__ == "__main__":
    # Update to point to JSONL files
    my_variance_files = {
        "Data-Dependent (Baseline)": "results/patch_test/baseline_ising/patch_variance_results.jsonl",
        "Layer-Wise (Lap 1)": "results/patch_test/lap_1/patch_variance_results.jsonl",
        "Layer-Wise (Lap 2)": "results/patch_test/lap_2/patch_variance_results.jsonl",
        "Layer-Wise (Lap 3)": "results/patch_test/lap_3/patch_variance_results.jsonl",
        "Layer-Wise (Lap 4)": "results/patch_test/lap_4/patch_variance_results.jsonl",
    }
    
    plot_variance_vs_radius(my_variance_files)