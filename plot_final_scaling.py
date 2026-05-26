import json
import matplotlib.pyplot as plt
import glob

def get_local_variance(folder_path):
    search_path = f"{folder_path}/patch_variance_results.jsonl"
    files = glob.glob(search_path)
    
    if not files: 
        print(f"  [X] Missing file: {search_path}")
        return None
        
    radii_vars = []
    try:
        with open(files[0], 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'radius_r' in data and 'var' in data:
                    radii_vars.append((float(data['radius_r']), float(data['var'])))
                    
        if radii_vars:
            # Sort by radius size 
            radii_vars.sort(key=lambda x: x[0])
            # Return the variance corresponding to the absolute smallest patch
            smallest_radius, corresponding_var = radii_vars[0]
            return corresponding_var
            
        # Fallback
        with open(files[0], 'r') as f:
            first_data = json.loads(f.readline())
            if 'var' in first_data:
                return float(first_data['var'])
                
    except Exception as e:
        print(f"  [!] Error reading JSON: {e}")
        
    return None

def plot_qubit_scaling():
    print("Gathering data for the final 25-qubit scaling plot...\n")
    
    qubit_sizes = [4, 9, 16, 25]
    
    id_vars = []
    dd_vars = []
    greedy_vars = []
    valid_q = []
    
    for n in qubit_sizes:
        print(f"Checking data for n = {n}...")
        
        base = f"results/final_scaling_lattice/n_{n}"
        
        v_id = get_local_variance(f"{base}/identity")
        v_dd = get_local_variance(f"{base}/standard_datadep")
        v_greedy = get_local_variance(f"{base}/greedy/true_initial_variance")
        
        print(f"  -> Identity: {v_id}")
        print(f"  -> Standard: {v_dd}")
        print(f"  -> Greedy:   {v_greedy}")
        
        if v_id is not None and v_dd is not None and v_greedy is not None:
            valid_q.append(n)
            id_vars.append(v_id)
            dd_vars.append(v_dd)
            greedy_vars.append(v_greedy)
            print(f"  [OK] Successfully loaded all data for n={n}\n")
        else:
            print(f"  [!] Skipping n={n} plot because it is missing data points.\n")
            
    if not valid_q:
        print("ERROR: No complete data points found. The graph will be blank!")
        return
        
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(valid_q, id_vars, marker='s', linewidth=2.5, color='orange', label="Identity (Baseline)")
    plt.plot(valid_q, dd_vars, marker='^', linewidth=2.5, color='red', label="Standard Data-Dependent")
    plt.plot(valid_q, greedy_vars, marker='o', linewidth=3.0, color='green', label="L4 - Layer-Wise")

    plt.yscale('log', base=10)
    plt.xticks(valid_q) 
    
    plt.title("Gradient Variance vs. Qubit Size (n) at Depth L=4", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Qubits (n)", fontsize=12)
    plt.ylabel("Local Gradient Variance (Log Scale)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    output_name = "lattice_25-2.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Success! Graph saved as '{output_name}'")

if __name__ == "__main__":
    plot_qubit_scaling()