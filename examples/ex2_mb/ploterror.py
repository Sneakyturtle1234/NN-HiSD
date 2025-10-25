import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

fontsize = 15
current_dir = os.path.dirname(os.path.abspath(__file__))
saddle_point = [-0.82200156, 0.6243128]
layer = [128, 128, 128]
num_points = [1000, 2000, 3000, 4000, 5000]
ratio = 0.0
steps = 200
k = 1
dt = 2e-3
w0 = np.array([-0.85, 0.65])
All_errors = []

grads_file_path = os.path.join(current_dir, 'parameters', 'nograds.txt')
errors_by_num_points = {num: [] for num in num_points}

with open(grads_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line and not line.startswith('#'): 
            try:
                num, error = line.split(',')
                num = int(num.strip())
                error = float(error.strip())
                if num in errors_by_num_points:
                    errors_by_num_points[num].append(error)
            except ValueError:
                print(f"Skip line: {line}")

for num in num_points:
    All_errors.append(errors_by_num_points[num])

labels = '1000', '2000', '3000', '4000', '5000'

max_values = [np.max(error) for error in All_errors]
min_values = [np.min(error) for error in All_errors]
q1_values = [np.percentile(error, 25) for error in All_errors]
q3_values = [np.percentile(error, 75) for error in All_errors]
median_values = [np.median(error) for error in All_errors]
mean_values = [np.mean(error) for error in All_errors]

fig, ax = plt.subplots()

for i in range(len(All_errors)):
    rect = patches.Rectangle((i+0.75, q1_values[i]), 0.5, q3_values[i]-q1_values[i], 
                             linewidth=1, edgecolor='k', facecolor='lightskyblue', alpha=0.5)
    ax.add_patch(rect)

    median_line = lines.Line2D([i+0.75, i+1.25], [median_values[i], median_values[i]], color='r')
    ax.add_line(median_line)

    min_line = lines.Line2D([i+1, i+1], [min_values[i], q1_values[i]], color='k')
    ax.add_line(min_line)
    max_line = lines.Line2D([i+1, i+1], [q3_values[i], max_values[i]], color='k')
    ax.add_line(max_line)


plt.plot(range(1, len(labels) + 1), max_values, marker='o', color='r', label='Max Error')
plt.plot(range(1, len(labels) + 1), mean_values, marker='o', color='b', label='Mean Error')
plt.plot(range(1, len(labels) + 1), min_values, marker='o', color='g', label='Min Error')

plt.xticks(range(1, len(labels) + 1), labels)
plt.ylim(bottom=1e-5, top=1e-1)
plt.xlabel('Training data amount', fontsize=15, labelpad=0)
plt.ylabel('Errors', fontsize=15, labelpad=0)
plt.legend(fontsize=15)
plt.grid(True)
plt.yscale('log')
plt.title('Potential values only', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig('Error_Picture.png', dpi=300)
plt.show()


All_errors = []
grads_file_path = os.path.join(current_dir, 'parameters', 'grads.txt')
errors_by_num_points = {num: [] for num in num_points}

with open(grads_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line and not line.startswith('#'): 
            try:
                num, error = line.split(',')
                num = int(num.strip())
                error = float(error.strip())
                if num in errors_by_num_points:
                    errors_by_num_points[num].append(error)
            except ValueError:
                print(f"Skip line: {line}")

for num in num_points:
    All_errors.append(errors_by_num_points[num])

labels = '1000', '2000', '3000', '4000', '5000'

max_values = [np.max(error) for error in All_errors]
min_values = [np.min(error) for error in All_errors]
q1_values = [np.percentile(error, 25) for error in All_errors]
q3_values = [np.percentile(error, 75) for error in All_errors]
median_values = [np.median(error) for error in All_errors]
mean_values = [np.mean(error) for error in All_errors]

fig, ax = plt.subplots()

for i in range(len(All_errors)):
    rect = patches.Rectangle((i+0.75, q1_values[i]), 0.5, q3_values[i]-q1_values[i], 
                             linewidth=1, edgecolor='k', facecolor='lightskyblue', alpha=0.5)
    ax.add_patch(rect)

    median_line = lines.Line2D([i+0.75, i+1.25], [median_values[i], median_values[i]], color='r')
    ax.add_line(median_line)

    min_line = lines.Line2D([i+1, i+1], [min_values[i], q1_values[i]], color='k')
    ax.add_line(min_line)
    max_line = lines.Line2D([i+1, i+1], [q3_values[i], max_values[i]], color='k')
    ax.add_line(max_line)


plt.plot(range(1, len(labels) + 1), max_values, marker='o', color='r', label='Max Error')
plt.plot(range(1, len(labels) + 1), mean_values, marker='o', color='b', label='Mean Error')
plt.plot(range(1, len(labels) + 1), min_values, marker='o', color='g', label='Min Error')

plt.xticks(range(1, len(labels) + 1), labels)
plt.ylim(bottom=1e-5, top=1e-1)
plt.xlabel('Training data amount',fontsize=15, labelpad=0)
plt.ylabel('Errors',fontsize=15,labelpad=0)
plt.legend(fontsize=15)
plt.grid(True)
plt.yscale('log')
plt.title('Potential and gradient values', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig('gradient.png', dpi=300)
plt.show()