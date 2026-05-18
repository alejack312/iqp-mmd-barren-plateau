import json
import glob
import os
import matplotlib.pyplot as plt

def plot_greedy_training():
    print("Gathering trajectory data...")
    plt.figure(figsize=(10, 6))
    
    # match number loops
    for lap in range(1, 5):
        search_path = f"results/patch_test/lap_{lap}/runs/*/trajectory.jsonl"
        trajectory_files = glob.glob(search_path)
        
        if not trajectory_files:
            print(f"-> Skipping Lap {lap}: No trajectory file found yet.")
            continue
            
        trajectory_file = max(trajectory_files, key=os.path.getmtime)
        
        steps = []
        losses = []
        
        # Open the JSONL file and extract the step number and loss valueq
        with open(trajectory_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                step = data.get('step', len(steps))
                loss = data.get('loss') or data.get('loss_total') or data.get('value')
                    
                if loss is not None:
                    steps.append(step)
                    losses.append(loss)
        
        if losses:
            plt.plot(steps, losses, label=f"{lap} Layer(s)", linewidth=2.5)

    plt.title("Layer-Wise Trainability: Loss vs. Circuit Depth", fontsize=14, fontweight='bold')
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("MMD Loss", fontsize=12)
    plt.legend(title="Circuit Depth", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the graph 
    output_filename = "trainability_scaling.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\nSuccess! Graph saved as '{output_filename}'")

if __name__ == "__main__":
    plot_greedy_training()