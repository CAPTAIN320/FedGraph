import os
import matplotlib.pyplot as plt
import numpy as np

model_accuracy = [0, 10, 25, 47, 76]
percentage_of_sybils = [0, 10, 20, 30, 40, 50]
num_of_epoch = [1, 5, 100, 500, 1000]
num_of_fake_edges = [0, 1, 10, 100, 1000, 10000]

os.makedirs('results', exist_ok=True)

### ATTACK 1
#1 plot_accuracy_per_epoch for varying percentage of sybils

#2 plot_accuracy_per_sybil_percentage at 1000 epochs



### ATTACK 2
#1a plot_accuracy_per_sybil_percentage if Random edge values

#1b plot_accuracy_per_sybil_percentage if Same edge values

#2 plot accuracy_per_number_of_fake_edges for varying percentage of sybils


