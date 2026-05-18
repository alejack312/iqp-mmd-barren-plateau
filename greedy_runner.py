import yaml
import subprocess
import os
import glob
import time

def run_greedy_training(max_layers=4, base_yaml_path="configs/experiments/scaling_ac_smoke.yaml"):
    print(f"Starting Greedy Layer-Wise Training up to {max_layers} layers...")
    
    for lap in range(1, max_layers + 1):
        print(f"\n{'='*50}")
        print(f" LAP {lap}: Training {lap} Layer(s)")
        print(f"{'='*50}")
        
        with open(base_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update the circuit depth
        config['circuit']['n_generators'] = f"{lap}*n"
        
        # Folder for each layer
        base_output = "results/patch_test"
        config['experiment']['output_dir'] = f"{base_output}/lap_{lap}"
        
        # Inject saved parameters
        if lap > 1:
            prev_lap = lap - 1
            
            search_path = f"{base_output}/lap_{prev_lap}/runs/*/checkpoints/step_*.npz"
            all_checkpoints = glob.glob(search_path)
            
            if not all_checkpoints:
                raise FileNotFoundError(f"Could not find checkpoints in Lap {prev_lap} folder!")
            
            latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
            print(f"-> Found previous foundation: {latest_checkpoint}")
            
            config['param_init_file'] = latest_checkpoint
            
        # Save temporary config 
        temp_yaml = f"configs/experiments/temp_greedy_L{lap}.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump(config, f)
            
        command = f"python -m iqp_bp.experiments.run_training --hyperparams {temp_yaml}"
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Lap {lap} completed successfully!")
            
            time.sleep(2) 
            
        except subprocess.CalledProcessError as e:
            print(f"Lap {lap} failed. Stopping the run.")
            break

if __name__ == "__main__":
    # Number layers can be chnaged
    run_greedy_training(max_layers=4)