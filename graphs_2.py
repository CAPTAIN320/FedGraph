import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('graphs', exist_ok=True)


datasets = ['citeseer']
num_sybils_array = [10, 20, 30, 40, 50]

for num_sybils in num_sybils_array:
    data = pd.read_csv(f'results_2/A2_citeseer_{num_sybils}-sybils_fake-edges.csv')
    num_fake_edges = data['num_fake_edges']
    accuracy = data['Accuracy']
    plt.plot(num_fake_edges, accuracy, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Fake Edges')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Fake Edges')
    plt.grid(True)
    plt.savefig(f'./graphs/A2_citeseer_{num_sybils}sybils_accuracy_num-of-fake-edges.png')
    plt.close()

