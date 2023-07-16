import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('graphs_2', exist_ok=True)

datasets = ['citeseer']

plt.figure()  # Create a new figure

##### ATTACK 1 ###########
num_sybils_array = [0]
for num_sybils in num_sybils_array:
    data = pd.read_csv(f'results_2/A1_citeseer_{num_sybils}_sybils.csv')
    epoch = data['Epoch']
    accuracy = data['Accuracy']
    label = f'{num_sybils}% sybils'
    plt.plot(epoch, accuracy, marker='x', linestyle='-', label=label)
plt.xlabel('Number of Fake Edges')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Fake Edges')
plt.grid(True)
plt.legend()  # Add legend based on the labels in the plot
plt.savefig('./graphs_2/A1_citeseer_sybils.png')
plt.close()

##### ATTACK 2 ###########
# A2_citeseer_accuracy_num-of-fake-edges
num_sybils_array = [10, 20, 30, 40, 50]
for num_sybils in num_sybils_array:
    data = pd.read_csv(f'results_2/A2_citeseer_{num_sybils}-sybils_fake-edges.csv')
    num_fake_edges = data['num_fake_edges']
    accuracy = data['Accuracy']
    label = f'{num_sybils}% sybils'
    plt.plot(num_fake_edges, accuracy, marker='o', linestyle='-', label=label)
plt.xlabel('Number of Fake Edges')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Fake Edges')
plt.grid(True)
plt.legend()  # Add legend based on the labels in the plot
plt.savefig('./graphs_2/A2_citeseer_accuracy_num-of-fake-edges.png')
plt.close()

# A2_citeseer_random-values
data = pd.read_csv(f'results_2/A2_citeseer_random-values.csv')
num_sybils = data['Sybil %']
accuracy = data['Accuracy']
plt.plot(num_sybils, accuracy, marker='o', linestyle='-', color='b')
plt.xlabel('Sybil %')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Sybil % random edge values')
plt.grid(True)
plt.ylim(0)
plt.savefig('./graphs_2/A2_citeseer_random-values.png')
plt.close()

# A2_citeseer_same-values
data = pd.read_csv(f'results_2/A2_citeseer_same-values.csv')
num_sybils = data['Sybil %']
accuracy = data['Accuracy']
plt.plot(num_sybils, accuracy, marker='o', linestyle='-', color='b')
plt.xlabel('Sybil %')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Sybil % same edge values')
plt.grid(True)
plt.ylim(0)
plt.savefig('./graphs_2/A2_citeseer_same-values.png')
plt.close()
