# Exploring Iris DataSet using Pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['species'] = iris.target_names[iris.target]
df.to_csv('iris_dataset.csv', index=False)
# Showing the dataset csv
print(df)

# Describe the data set.
df.describe()

# Checking missing values
df.isnull().sum()

# Counts the difirent species.

df['species'].value_counts()

# Histogram

plt.figure(figsize=(10, 7))
x = iris.data[:, 0]  # Sepal Length
plt.hist(x, bins=20, color="green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 7))
x = iris.data[:, 1]  # Sepal Width
plt.hist(x, bins=20, color="black")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 7))
x = iris.data[:, 2]  # Petal Length
plt.hist(x, bins=20, color="blue")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 7))
x = iris.data[:, 3]  # Petal Width
plt.hist(x, bins=20, color="red")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Count")
plt.show()

# Create subplots for all four species
plt.figure(figsize=(10, 7))
x = iris.data[:, 0]  # Sepal Length
plt.hist(x, bins=20, color="green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")

x = iris.data[:, 1]  # Sepal Width
plt.hist(x, bins=20, color="black")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Count")

x = iris.data[:, 2]  # Petal Length
plt.hist(x, bins=20, color="blue")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")

x = iris.data[:, 3]  # Petal Width
plt.hist(x, bins=20, color="red")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Count")
plt.show()

# Extract features (sepal length and sepal width)
X = iris.data[:, :2]
y = iris.target

# Create a scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')

# Add labels and title
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Species Scatter Plot")

# Show the plot
plt.show()
