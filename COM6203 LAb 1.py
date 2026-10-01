import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statistics import variance, stdev

# Part 1: Why Statistics Matters
print("Statistics helps us understand data, patterns, uncertainty and model performance.\n")


# Part 2: Population vs Sample
print("Population = full dataset, Sample = subset used for analysis.\n")

# Part 3: Mean, Median, Mode-
data = np.array([4,5,1,2,7,2,6,9,3])
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", stats.mode(data).mode)

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
print("\nCar Speed Stats:")
print("Mean:", np.mean(speed))
print("Median:", np.median(speed))
print("Mode:", stats.mode(speed).mode)


# Part 4: Outliers & Boxplot
delivery_times = [12,15,14,16,15,14,13,14,120]
plt.boxplot(delivery_times)
plt.title("Delivery Times Boxplot")
plt.show()

print("Outlier affects mean more because mean uses ALL values, median only uses the middle.\n")


# Part 5: Variance & Std Dev
game_points = [35,56,43,59,63,79,35,41,64,43,93,60,77,24,82]
print("Variance:", variance(game_points))
print("Standard Deviation:", stdev(game_points))


# Part 6: Percentiles & Quartiles
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
print("\nPercentiles (25,50,75,90):", np.percentile(ages, [25,50,75,90]))


# Part 7: IQR Outlier Detection
transactions = [25,30,18,27,29,24,31,22,4500]
Q1 = np.percentile(transactions, 25)
Q3 = np.percentile(transactions, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nIQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)
print("Outliers:", [x for x in transactions if x < lower or x > upper])


# Part 8: Correlation
hours_studied = [1,2,3,4,5,6,7]
exam_scores = [45,50,58,62,70,81,90]
print("\nCorrelation Matrix:")
print(np.corrcoef(hours_studied, exam_scores))


# Part 9: Normal Distribution
x = np.random.normal(50, 10, 10000)
plt.hist(x, bins=50)
plt.title("Normal Distribution")
plt.show()


# Part 10: Hypothesis Testing
xbar = 990
mu0 = 1000
s = 12.5
n = 30
t = (xbar - mu0) / (s / np.sqrt(n))
print("\nT-statistic:", t)


# Part 11: Type I vs Type II Errors

print("\nFalse Negative is worse in cancer detection because real cancer goes untreated.\n")


# Part 12: Confusion Matrix
TP, TN, FP, FN = 80, 170, 20, 30
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy:", accuracy)


# Part 13: Entropy
print("\n90/10 coin has lower entropy because outcomes are more predictable.\n")


# Part 14: Synthetic Data
x = np.random.uniform(0, 5, 250)
plt.hist(x, bins=5)
plt.title("Uniform Distribution")
plt.show()

# Part 15: Mini Project (Simple Example)
# Fake dataset
attendance = np.array([80,85,90,70,60,95,88,92])
study_hours = np.array([2,3,4,1,1,5,4,6])
coursework = np.array([65,70,75,55,50,80,78,85])
performance = np.array([68,72,80,60,55,88,82,90])

df = pd.DataFrame({
    "attendance": attendance,
    "study_hours": study_hours,
    "coursework": coursework,
    "performance": performance
})

print("\nMini Project Data:")
print(df)

print("\nCorrelations:")
print(df.corr())

# Simple linear regression
from sklearn.linear_model import LinearRegression

X = df[["attendance", "study_hours", "coursework"]]
y = df["performance"]

model = LinearRegression()
model.fit(X, y)

print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Predictions:", model.predict(X))
