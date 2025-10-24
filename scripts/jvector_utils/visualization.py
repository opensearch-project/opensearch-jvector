"""
Visualization utilities for JVector performance metrics.

This module provides plotting functions for analyzing merge times,
quantization performance, and other JVector metrics.
"""

import csv
import os
import matplotlib.pyplot as plt


def plot_merge_times(csv_file):
    """Plot graph merge time and quantization training time vs number of documents"""
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return
    
    # Read CSV data
    num_docs = []
    graph_merge_times = []
    quantization_times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_docs.append(int(row['num_documents']))
            graph_merge_times.append(int(row['graph_merge_time_ms']))
            quantization_times.append(int(row['quantization_training_time_ms']))
    
    if not num_docs:
        print("No data found in CSV file")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graph merge time plot
    ax1.plot(num_docs, graph_merge_times, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Documents')
    ax1.set_ylabel('Graph Merge Time (ms)')
    ax1.set_title('JVector Graph Merge Time vs Number of Documents')
    ax1.grid(True, alpha=0.3)
    
    # Quantization training time plot
    ax2.plot(num_docs, quantization_times, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Documents')
    ax2.set_ylabel('Quantization Training Time (ms)')
    ax2.set_title('JVector Quantization Training Time vs Number of Documents')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    # Show plot
    plt.show()

