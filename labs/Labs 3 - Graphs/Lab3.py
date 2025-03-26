import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Given data
data = np.array([
    [28, 175, 78, 10000, 10000],
    [34, 162, 43, 70000, 5000],
    [40, 180, 85, 90000, 125000],
    [26, 168, 60, 45000, 8000],
    [50, 197, 95, 110000, 30000],
    [31, 160, 55, 60000, 12000],
    [45, 182, 88, 95000, 25000],
    [29, 170, 70, 190000, 15000],
    [38, 178, 83, 85000, 18000],
    [82, 145, 58, 40000, 7000]
])

# 0-1 Normalization
scaler = MinMaxScaler()
data_0_1_normalized = scaler.fit_transform(data)

# Mean Normalization
data_mean_normalized = (data - np.mean(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Z-score Normalization
scaler = StandardScaler()
data_z_score_normalized = scaler.fit_transform(data)

# Check for outliers using Z-score method
z_scores = np.abs(data_z_score_normalized)
outliers = np.where(z_scores > 3)

# Print results
print("0-1 Normalized Data:\n", data_0_1_normalized)
print("\nMean Normalized Data:\n", data_mean_normalized)
print("\nZ-score Normalized Data:\n", data_z_score_normalized)
print("\nOutliers (row, column):\n", list(zip(outliers[0], outliers[1])))


import numpy as np
import matplotlib.pyplot as plt

# Data for each line
lines = [
    {"Left": {"Stars": 7, "Circles": 5}, "Right": {"Stars": 18, "Circles": 20}},
    {"Left": {"Stars": 17, "Circles": 6}, "Right": {"Stars": 8, "Circles": 19}},
    {"Left": {"Stars": 24, "Circles": 16}, "Right": {"Stars": 1, "Circles": 9}}
]

# Function to compute metrics
def compute_metrics(line):
    TP = line["Left"]["Stars"]  # Stars correctly classified as stars
    FP = line["Left"]["Circles"]  # Circles incorrectly classified as stars
    FN = line["Right"]["Stars"]  # Stars incorrectly classified as circles
    TN = line["Right"]["Circles"]  # Circles correctly classified as circles

    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0
    Accuracy = (TP + TN) / (TP + FP + TN + FN)

    return Precision, Recall, F1_Score, Accuracy

# Compute metrics for each line
metrics = []
for i, line in enumerate(lines):
    Precision, Recall, F1_Score, Accuracy = compute_metrics(line)
    metrics.append((Precision, Recall, F1_Score, Accuracy))
    print(f"Line {i+1}: Precision = {Precision:.2f}, Recall = {Recall:.2f}, F1-Score = {F1_Score:.2f}, Accuracy = {Accuracy:.2f}")

# Plot Precision-Recall Curve with switched axes
plt.figure(figsize=(8, 6))
for i, line in enumerate(lines):
    Precision, Recall, _, _ = compute_metrics(line)
    plt.scatter(Precision, Recall, label=f"Line {i+1} (P={Precision:.2f}, R={Recall:.2f})")  # Switched axes

plt.xlabel("Precision")  # Precision on x-axis
plt.ylabel("Recall")  # Recall on y-axis
plt.title("Precision-Recall Curve (Switched Axes)")
plt.legend()
plt.grid()
plt.show()