import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    results_path = 'results/final_results.csv'
    if not os.path.exists(results_path):
        print("No results found.")
        return

    df = pd.read_csv(results_path)
    
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Accuracy Comparison per Domain
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='dataset', y='accuracy', hue='model')
    plt.title('Accuracy Comparison per Domain')
    plt.ylim(0, 1.0)
    plt.savefig('results/plots/accuracy_comparison.png')
    plt.close()
    
    # 2. Accuracy vs Efficiency (Params)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='params', y='accuracy', hue='dataset', style='model', s=100)
    plt.title('Accuracy vs Parameter Efficiency')
    plt.xscale('log')
    plt.savefig('results/plots/accuracy_vs_efficiency.png')
    plt.close()

    print("Plots generated in results/plots/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    
    if args.plot_only:
        plot_results()
        return

if __name__ == "__main__":
    main()
