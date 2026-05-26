# Uniform Initialization Baseline: Gradient Variance vs. Qubit Size

import yaml
import subprocess
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def run_uniform_scaling():
    base_yaml_path = "configs/experiments/scaling_ac_smoke.yaml"
    qubit_sizes = [4, 9, 16, 25]
    run_name = "Uniform_Lattice"
    results_dict = {"n_qubits": [], "gradient_variance": []}
    
    print(f"{'='*60}")
    print(f" STARTING UNIFORM INITIALIZATION BASELINE")
    print(f"{'='*60}")

    # Ensure the base config exists
    if not os.path.exists(base_yaml_path):
        print(f"ERROR: Could not find base config at {base_yaml_path}")
        return

    with open(base_yaml_path, 'r') as f:
        base_config = yaml.safe_load(f)

    if 'erdos_renyi' in base_config.get('circuit', {}):
        del base_config['circuit']['erdos_renyi']

    for n in qubit_sizes:
        print(f"\n--- Calculating Variance for n = {n} Qubits ---")
        
        output_dir = f"results/{run_name}/n_{n}"
        
        base_config['experiment']['output_dir'] = output_dir
        base_config['circuit']['family'] = ['lattice']
        base_config['circuit']['n_qubits'] = [n]
        base_config['init']['scheme'] = ['uniform']
        
        temp_yaml = f"configs/experiments/temp_uniform_n{n}.yaml"
        os.makedirs(os.path.dirname(temp_yaml), exist_ok=True)
        with open(temp_yaml, 'w') as f:
            yaml.dump(base_config, f)
            
        subprocess.run(f"python -m iqp_bp.experiments.run_patch_variance --hyperparams {temp_yaml}", shell=True)
        
        jsonl_path = os.path.join(output_dir, "patch_variance_results.jsonl")
        
        if os.path.exists(jsonl_path):
            variances = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'var' in data:
                            variances.append(data['var'])
            
            if variances:
                # Average the variance across all patch radii
                mean_variance = np.mean(variances)
                results_dict["n_qubits"].append(n)
                results_dict["gradient_variance"].append(mean_variance)
                print(f" -> Success! Mean Gradient Variance for n={n}: {mean_variance:.2e}")
            else:
                print(f" -> WARNING: Could not find 'var' keys in {jsonl_path}.")
        else:
            print(f" -> ERROR: No JSONL output file found at {jsonl_path}.")


    # Generate Plot
    if len(results_dict["n_qubits"]) > 0:
        import matplotlib.ticker as ticker
        
        plt.figure(figsize=(8, 6), dpi=300)
        
        n_vals = np.array(results_dict["n_qubits"])
        var_vals = np.array(results_dict["gradient_variance"])
        
        plt.plot(
            n_vals, 
            var_vals, 
            marker='s', 
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=1.5,
            linestyle='--', 
            color='#d62728', 
        )
        
        plt.yscale('log')
        plt.xticks(n_vals, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Make the grid look professional
        plt.grid(True, which="major", ls="-", alpha=0.2, color='gray')
        plt.grid(True, which="minor", ls=":", alpha=0.1, color='gray')
        
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            
        plt.xlabel(r'Number of Qubits ($n$)', fontsize=16)
        plt.ylabel(r'$\mathrm{Var}_\theta[\partial C(\theta)]$', fontsize=16)
        
        plt.title('Gradient variance scaling under Uniform Initialization', fontsize=16, pad=5)
        
        plot_path = f"results/{run_name}/uniform_var.png"
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        print(f"[+] Clean plot successfully saved to: {plot_path}")
        plt.show()

if __name__ == "__main__":
    run_uniform_scaling()