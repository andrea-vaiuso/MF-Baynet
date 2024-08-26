import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

low_fid_df = pd.read_csv("Dataset/Mach_AoA_CL_CM_lf.csv")
mid_fid_df = pd.read_csv("Dataset/Mach_AoA_CL_CM_mf.csv")
hig_fid_df = pd.read_csv("Dataset/Mach_AoA_CL_CM_hf.csv")

mid_fid_df = mid_fid_df.drop([28, 22])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111) #, projection='3d')

df1 = low_fid_df
df2 = mid_fid_df
df3 = hig_fid_df
z = "cl"

ax.scatter(df1["mach"],df1["aoa"], color='blue', alpha=0.4, s=5, label="Low Fidelity")
ax.scatter(df2["mach"],df2["aoa"], color='orange', alpha=0.7, s=40, label="Mid Fidelity")
ax.scatter(df3["mach"], df3["aoa"], color='red', s=50, label='High Fidelity')
ax.set_xlabel("Mach")
ax.set_ylabel("AoA")

# ax.scatter(df1["mach"],df1["aoa"],df1[z])
# ax.scatter(df2["mach"],df2["aoa"],df2[z])
# ax.scatter(df3["mach"], df3["aoa"], df3[z], color='red', label='High Fidelity Points')
# ax.set_xlabel("Mach")
# ax.set_ylabel("AoA")
# ax.set_zlabel("CL")
plt.legend()
plt.show()
