import pandas as pd
import matplotlib.pyplot as plt
from supervised.training import training


def visualization():
    results_df = pd.DataFrame(training())
    results_df.to_csv("data/model_comparison_results.csv", index=False)
    results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis='y')    
    plt.savefig("data/model_performance.png")
    plt.close() 
    

