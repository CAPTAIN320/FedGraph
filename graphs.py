import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

num_sybils_array = [0, 10, 20, 30, 40, 50]

num_fake_edges = 10
datasets = ['citeseer', 'cora', 'pubmed']

### ATTACK 1
#1 plot_accuracy_over_epoch for varying percentage of sybils
for index, dataset in enumerate(datasets):
    for num_sybils in num_sybils_array:
        name = f'A1_{dataset}_{num_sybils}_sybils'
        data = pd.read_csv(f'results/{name}.csv')
        epochs = data['Epoch']
        accuracy = data['Accuracy']
        plt.plot(epochs, accuracy)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs {num_sybils}% Sybils')
    plt.savefig(f'./graphs/{name}.png')
    plt.close()


#2 plot_accuracy_per_sybil_percentage at 10 epochs
for index, dataset in enumerate(datasets):
    for num_sybils in num_sybils_array:
        name = f'A2_{dataset}_{num_sybils}_sybils_{num_fake_edges}_same-values'
        data = pd.read_csv(f'results/{name}.csv')
        epochs = data['Sybil %']
        accuracy = data['Accuracy']
        plt.plot(epochs, accuracy)
        plt.xlabel('Sybil %')
        plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Sybil % {num_sybils}% Sybils')
    plt.savefig(f'./graphs/{name}.png')
    plt.close()

# plt.plot(epochs, accuracy)
# model_accuracy = [0, 10, 25, 47, 76]
# percentage_of_sybils = [0, 10, 20, 30, 40, 50]
# num_of_epoch = [1, 5, 100, 500, 1000]
# num_of_fake_edges = [0, 1, 10, 100, 1000, 10000]

# os.makedirs('graphs', exist_ok=True)

### ATTACK 1
#1 plot_accuracy_per_epoch for varying percentage of sybils

#2 plot_accuracy_per_sybil_percentage at 1000 epochs



### ATTACK 2
#1a plot_accuracy_per_sybil_percentage if Random edge values

#1b plot_accuracy_per_sybil_percentage if Same edge values

#2 plot accuracy_per_number_of_fake_edges for varying percentage of sybils


