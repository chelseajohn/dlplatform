import pandas as pd
import os
import matplotlib.pyplot as plt


path = "U:/Desktop/federatedLearning/MNISTtorchCNN_2020-07-27_20-27-12/" ####### path to  folder

## extracting the number of workers
expSummary = open(os.path.join(path,"summary.txt")).read()

for l in expSummary.split('\n'):
    if "Number of Nodes:" in l:
        nodesNumber = int(l.split("\t")[-1])
print("Number of nodes=", nodesNumber)

#### plotting the lossValue of workers
for worker in range(nodesNumber):
    worker_path = os.path.join(path, f"worker{worker}/losses.txt")
    data = pd.read_csv(worker_path, sep="\s+")
    time = data['Time']  # in sec
    loss = data['LossValue']
    plt.scatter(time-time[0], loss, label=f'worker{worker}') # plot against relative time

plt.title("Loss Value of Workers")
plt.xlabel(" Relative Time in sec ")
plt.ylabel("Loss Value")
plt.legend()
plt.grid()
plt.savefig(os.path.join(path, "Lossplot.pdf"))
plt.show()






