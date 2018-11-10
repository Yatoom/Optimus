import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rfr = pd.read_csv("rfr-frame.csv", index_col=0)
lgbm = pd.read_csv("lgbm-frame.csv", index_col=0)

# Plot RFR
std = rfr.std(axis=1)
mean = rfr.mean(axis=1)
plt.plot(mean, color="red")
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color="red", label="SMAC-RF EI")

# Plot LGBM
std = lgbm.std(axis=1)
mean = lgbm.mean(axis=1)
plt.plot(mean, color="blue")
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color="blue", label='LightGBM QR')
plt.legend()

plt.xlabel("Iteration")
plt.ylabel("Acumulated maximum as fraction of optimum")

plt.show()