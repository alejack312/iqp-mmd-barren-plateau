"""
Mass Data Generator for Patch Radius vs. Gradient Variance.
Generates data for:
- Datasets: Ising, Blobs, Genomic
- Initializations: Uniform (Identity), Data-Dependent
- Qubit Sizes (n): 9, 16, 25
"""

import yaml
import subprocess
import os

def generate_all_data():
    base_yaml_path = "configs/experiments/scaling_ac_smoke.yaml"
    
    datasets = ['product_bernoulli', 'binary_mixture', 'ising']
    initializations = ['identity', 'data_dependent']
    qubit_sizes = [9, 16, 25]
    
    print(f"{'='*60}")
    print(" STARTING MASS DATA GENERATION FOR 3-PANEL GRAPH")
    print(f" Total runs queued: {len(datasets) * len(initializations) * len(qubit_sizes)}")
    print(f"{'='*60}")

    if not os.path.exists(base_yaml_path):
        print(f"ERROR: Could not find base config at {base_yaml_path}")
        return

    with open(base_yaml_path, 'r') as f:
        base_config = yaml.safe_load(f)

    run_counter = 1

    for ds in datasets:
        for init in initializations:
            for n in qubit_sizes:
                print(f"\n[{run_counter}/18] Running -> Dataset: {ds.upper()} | Init: {init.upper()} | n={n}")
                
                # Setup specific output directory so JSONL files don't overwrite each other
                output_dir = f"results/mass_patch_data/{ds}/{init}/n_{n}"
                
                # Override parameters
                base_config['experiment']['output_dir'] = output_dir
                base_config['dataset']['type'] = ds
                base_config['init']['scheme'] = [init]
                base_config['circuit']['n_qubits'] = [n]
                
                # Write temporary YAML
                temp_yaml = f"configs/experiments/temp_{ds}_{init}_n{n}.yaml"
                os.makedirs(os.path.dirname(temp_yaml), exist_ok=True)
                with open(temp_yaml, 'w') as f:
                    yaml.dump(base_config, f)
                    
                # Run the patch variance experiment
                try:
                    subprocess.run(
                        f"python -m iqp_bp.experiments.run_patch_variance --hyperparams {temp_yaml}", 
                        shell=True, 
                        check=True
                    )
                except subprocess.CalledProcessError:
                    print(f" [!] ERROR during {ds} - {init} - n={n}. Continuing to next run...")
                
                run_counter += 1

    print(f"\n{'='*60}")
    print(" ALL DATA GENERATION COMPLETE.")
    print(" You can now run the plotting script.")
    print(f"{'='*60}")

    # Clean up the last temp file
    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

if __name__ == "__main__":
    generate_all_data()