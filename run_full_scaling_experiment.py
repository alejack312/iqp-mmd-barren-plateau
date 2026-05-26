import yaml
import subprocess
import glob
import os

def run_experiment():
    base_yaml = "configs/experiments/scaling_ac_smoke.yaml"
    qubit_sizes = [4, 9, 16, 25] 
    max_layers = 4
    
    run_name = "lattice" 
    
    for n in qubit_sizes:
        print(f"\n{'='*60}")
        print(f" STARTING EXPERIMENT FOR n = {n} QUBITS | {run_name.upper()}")
        print(f"{'='*60}")
        
        with open(base_yaml, 'r') as f:
            config = yaml.safe_load(f)
            
        config['circuit']['n_qubits'] = [n]
        config['dataset']['type'] = 'ising' # Fixed to match your YAML
        temp_yaml = f"configs/experiments/temp_n{n}.yaml"
        
        base_output_dir = f"results/final_scaling_{run_name}/n_{n}"
        
        # BASELINE: Identity Variance
        print(f"\n--- [1/3] Calculating 4-Layer Identity Baseline (n={n}) ---")
        config['circuit']['n_generators'] = f"{max_layers}*n"
        config['init']['scheme'] = ['small_angle']
        if 'param_init_file' in config: del config['param_init_file']
        config['experiment']['output_dir'] = f"{base_output_dir}/identity"
        
        with open(temp_yaml, 'w') as f: yaml.dump(config, f)
        subprocess.run(f"python -m iqp_bp.experiments.run_patch_variance --hyperparams {temp_yaml}", shell=True)

        # BASELINE: Data-Dependent Variance
        print(f"\n--- [2/3] Calculating 4-Layer Standard Data-Dependent (n={n}) ---")
        config['init']['scheme'] = ['data_dependent']
        config['experiment']['output_dir'] = f"{base_output_dir}/standard_datadep"
        
        with open(temp_yaml, 'w') as f: yaml.dump(config, f)
        subprocess.run(f"python -m iqp_bp.experiments.run_patch_variance --hyperparams {temp_yaml}", shell=True)

        # LAYER-WISE
        print(f"\n--- [3/3] Running Greedy Layer-Wise Training (n={n}) ---")
        current_checkpoint = None
        
        for lap in range(1, max_layers):
            print(f"  -> Training Lap {lap}...")
            config['circuit']['n_generators'] = f"{lap}*n"
            config['experiment']['output_dir'] = f"{base_output_dir}/greedy/lap_{lap}"
            
            if current_checkpoint:
                config['param_init_file'] = current_checkpoint
            elif 'param_init_file' in config:
                del config['param_init_file']
                
            with open(temp_yaml, 'w') as f: yaml.dump(config, f)
            
            # RUN TRAINING
            subprocess.run(f"python -m iqp_bp.experiments.run_training --hyperparams {temp_yaml}", shell=True)
            
            checkpoints = glob.glob(f"{base_output_dir}/greedy/lap_{lap}/runs/*/checkpoints/step_*.npz")
            if checkpoints:
                current_checkpoint = max(checkpoints, key=os.path.getmtime)
            else:
                print(f"Failed to find checkpoint for Lap {lap}. Aborting n={n}.")
                break
                
        # MEASURE initial VARIANCE OF LAP 4
        if current_checkpoint:
            print(f"  -> Calculating True Initial Variance for 4-Layer Greedy Model...")
            config['circuit']['n_generators'] = f"{max_layers}*n" 
            config['experiment']['output_dir'] = f"{base_output_dir}/greedy/true_initial_variance"
            config['param_init_file'] = current_checkpoint        
            
            with open(temp_yaml, 'w') as f: yaml.dump(config, f)
            
            subprocess.run(f"python -m iqp_bp.experiments.run_patch_variance --hyperparams {temp_yaml}", shell=True)

if __name__ == "__main__":
    run_experiment()