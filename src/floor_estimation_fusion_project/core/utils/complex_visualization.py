"""
script for visualize rusults from detection
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# define path to BRE-TF FILE
main_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__))))))

folder = os.path.join(main_dir, r'results\\floor_detection_results\\LESNA\\', os.path.basename("results_data.csv"))


major_ticks = np.arange(-12, 13, 1)
major_ticks_x = np.arange(0, 1000, 20)
minor_ticks = np.arange(0, 1000, 10)

plt.rcParams['figure.figsize'] = [15, 8]
new_data = np.genfromtxt(folder, delimiter='\t')



# data z fuze
time = new_data[1:, 0]
camera = new_data[1:, 1]
acc = np.absolute(new_data[1:, 2])
fusion = new_data[1:, 3]
bayes = new_data[1:, 4]
true_floor = new_data[1:, 5]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
rows = ['{}'.format(row) for row in ['floor number \n[-]', "floor number \n[-]"]]

for ax, row in zip(axs, rows):
    ax.set_ylabel(row)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks_x)
    ax.grid(which='both', linestyle='--')

axs[0].scatter(time, acc, s=100, marker="v")
axs[0].scatter(time, camera, marker="p")
axs[0].legend(["diff floor - acc", "disp floor - cam"], fontsize='medium')
axs[0].set_title("Accelerometer and camera algorithm results")

axs[1].scatter(time, fusion, s=100, marker="v")
axs[1].scatter(time, true_floor, marker="p")
axs[1].set_xlabel("time [s]")
axs[1].legend(["fusion result", "reality"], fontsize='medium')
axs[1].set_title("Fusion result vs. reality")
plt.show()
