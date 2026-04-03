import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_plots(suffix, output_dir, threshold=0.18):
    csv_path = os.path.join(output_dir, f'academic_results_{suffix}.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Generating plots for {suffix} samples...")

    # 1. Topic Vulnerability Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Topic', y='NLI_Score', data=df, palette='vlag', hue='Topic', legend=False)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Hallucination Threshold')
    plt.title(f"Domain Vulnerability Mapping - Size {suffix}")
    plt.xticks(rotation=45)
    plt.ylabel("Inconsistency Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_{suffix}.png'))
    plt.close()

    # 2. Distribution Curves (KDE Separation)
    plt.figure(figsize=(10, 6))
    scores_fact = df[df['TrueLabel'] == 0]['NLI_Score']
    scores_hallu = df[df['TrueLabel'] == 1]['NLI_Score']
    
    if not scores_fact.empty: sns.kdeplot(scores_fact, fill=True, label="Factual", color="green")
    if not scores_hallu.empty: sns.kdeplot(scores_hallu, fill=True, label="Hallucinated", color="red")
    
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.title(f"Engine Confidence Density - Size {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_{suffix}.png'))
    plt.close()

    # 3. Domain Risk Heatmap
    plt.figure(figsize=(14, 4)) # Increased width for better label clearance
    avg_vuln = df.groupby('Topic')['NLI_Score'].mean().to_frame().T
    sns.heatmap(avg_vuln, annot=True, cmap='Reds', fmt='.2f', cbar=False) # Simplified for clarity
    plt.title(f"Domain-Level Hallucination Risk Heatmap - Size {suffix}", pad=20)
    plt.xlabel("") # Clearer x-label
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{suffix}.png'), bbox_inches='tight') # High fidelity bounds
    plt.close()

if __name__ == "__main__":
    # Explicit path based on user environment
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "final idea", "Results")
    
    for size in ["50", "100", "200"]:
        generate_plots(size, work_dir)
    
    print("\nVisualizations generated successfully!")
