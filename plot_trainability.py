import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_greedy_training():
    print("Gathering trajectory data...")
    
    # Set professional journal-style parameters
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm"
    })
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Use a sequential colormap to show layer progression clearly
    colors = cm.get_cmap('viridis', 5)
    
    for lap in range(1, 5):
        search_path = f"results/patch_test/lap_{lap}/runs/*/trajectory.jsonl"
        trajectory_files = glob.glob(search_path)
        
        if not trajectory_files:
            print(f"-> Skipping Lap {lap}: No trajectory file found.")
            continue
            
        trajectory_file = max(trajectory_files, key=os.path.getmtime)
        
        steps = []
        losses = []
        
        with open(trajectory_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Handle your variable naming conventions
                    step = data.get('step', len(steps))
                    loss = data.get('loss') or data.get('loss_total') or data.get('value')
                    
                    if loss is not None:
                        steps.append(step)
                        losses.append(loss)
                except json.JSONDecodeError:
                    continue
        
        if losses:
            ax.plot(
                steps, losses, 
                label=f"Layer $L={lap}$", 
                linewidth=2.0, 
                color=colors(lap-1),
                alpha=0.9
            )

    # Styling for professional publication
    ax.set_title("Layer-Wise Trainability: Loss Convergence", fontsize=16, pad=15)
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("MMD Loss", fontsize=14)
    
    # Scale and Grids
    ax.set_yscale('log')
    ax.grid(True, which="major", ls="-", alpha=0.2, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.1, color='gray')
    
    # Professional borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)

    # Legend
    ax.legend(fontsize=12, frameon=True, edgecolor='black')
    
    # Save the graph
    output_filename = "trainability_professional.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    
    print(f"\n[+] Success! Professional graph saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    plot_greedy_training()