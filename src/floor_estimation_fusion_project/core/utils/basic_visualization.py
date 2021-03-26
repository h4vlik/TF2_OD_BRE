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

folder = os.path.join(main_dir, os.path.basename("results_data.csv"))
folder = os.path.join(main_dir, r"data\\Acc_input_data\\lesna_01", os.path.basename("lesna_01.csv"))

plt.rcParams['figure.figsize'] = [15, 8]
new_data = np.genfromtxt(folder, delimiter='\t')

# data z fuze
time = new_data[1:, 0]
camera = new_data[1:, 1]
#acc = np.absolute(new_data[1:, 2])
#fusion = new_data[1:, 3]


plt.plot(time, camera)
#plt.plot(time, acc)
#plt.plot(time, fusion)
#plt.legend(["camera", "acc", "fusion"])
plt.title('Visualization of DATA FUSION')
plt.show()