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
    plt.figure()  # Create a new figure for each dataset
    lines = []  # Store the lines for creating the legend

    for num_sybils in num_sybils_array:
        name = f'A1_{dataset}_{num_sybils}_sybils'
        data = pd.read_csv(f'results/{name}.csv')
        epochs = data['Epoch']
        accuracy = data['Accuracy']
        line, = plt.plot(epochs, accuracy)
        lines.append(line)  # Add the line to the legend

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

    # Create the legend with labels
    labels = [f'{num_sybils}% Sybils' for num_sybils in num_sybils_array]
    plt.legend(lines, labels)

    plt.title(f'Accuracy over Epochs - {dataset}')
    plt.savefig(f'./graphs/{dataset}_accuracy.png')
    plt.close()


#2 plot_accuracy_per_sybil_percentage at 10 epochs
for index, dataset in enumerate(datasets):
    plt.figure()  # Create a new figure for each dataset
    lines = []  # Store the lines for creating the legend

    for num_sybils in num_sybils_array:
        name = f'A2_{dataset}_{num_sybils}_sybils_{num_fake_edges}_same-values'
        data = pd.read_csv(f'results/{name}.csv')
        sybils = data['Sybil %']
        accuracy = data['Accuracy']
        line, = plt.plot(sybils, accuracy)
        lines.append(line)  # Add the line to the legend

        plt.xlabel('Sybil %')
        plt.ylabel('Accuracy')

    # Create the legend with labels
    labels = [f'{num_sybils}% Sybils' for num_sybils in num_sybils_array]
    plt.legend(lines, labels)

    plt.title(f'Accuracy over Sybil % - {dataset}')
    plt.savefig(f'./graphs/{dataset}_sybil_accuracy.png')
    plt.close()




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
        sybils = data['Sybil %']
        accuracy = data['Accuracy']
        plt.plot(sybils, accuracy)
        plt.xlabel('Sybil %')
        plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Sybil % {num_sybils}% Sybils')
    plt.savefig(f'./graphs/{name}.png')
    plt.close()
