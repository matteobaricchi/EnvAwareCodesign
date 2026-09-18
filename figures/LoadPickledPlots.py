#%% Preamble
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np

#%% Load and open figures
for i in np.arange(1,17):
    # first try to open the normal figure
    try:
        with open(f"fig{i}\\fig{i}.pkl", "rb") as f:
            fig = pickle.load(f)
        plt.show()
    except:
        # Then try a, b, c
        for suffix in ["a", "b", "c"]:
            try:
                with open(f"fig{i}\\fig{i}{suffix}.pkl", "rb") as f:
                    fig = pickle.load(f)
                plt.show()
            except:
                break

#%% Example to read some data from figure
# with open("fig10\\fig10.pkl", "rb") as f:
#     fig = pickle.load(f)
# plt.show()

# # Load and print scattered data of first subplot
# ax = fig.axes[0]
# for collection in ax.collections:
#     if isinstance(collection, PathCollection):
#         offsets = collection.get_offsets()

#         x = np.asarray(offsets[:, 0])
#         y = np.asarray(offsets[:, 1])

#         print("x:", x)
#         print("y:", y)