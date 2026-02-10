import numpy as np

orig = np.load("/Users/simon/Desktop/predictions_official.npz")["arr_0"]
reim = np.load("/Users/simon/Desktop/predictions_reimpl.npz")["arr_0"]

print(f"Official shape: {orig.shape}")
print(f"Reimplementation shape: {reim.shape}")

print(f"Official - Mean: {np.mean(orig)}")
print(f"Official - Max: {np.max(orig)}")
print(f"Official - Min: {np.min(orig)}")

print(f"Reimplementation - Mean: {np.mean(reim)}")
print(f"Reimplementation - Max: {np.max(reim)}")
print(f"Reimplementation - Min: {np.min(reim)}")

print(f"Maximum absolute difference: {np.max(np.abs(orig - reim))}")

correlation = np.corrcoef(orig.flatten(), reim.flatten())[0, 1]
print(f"Pearson correlation coefficient: {correlation}")
