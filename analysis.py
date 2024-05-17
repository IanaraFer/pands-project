# Exploring Iris DataSet using Pandas
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA

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

# Variables plots
df = sns.load_dataset('iris')
print(df.head())

# Get just the petal lengths#
plen = df['petal_length']

# Types
print(type(plen))

# Just get teh numpy array.
plen = plen.to_numpy()

# Show
plen

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

# Iris plot
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names,
    loc="lower right", title="Classes")

# unused but required import for doing 3d projections with matplotlib < 3.2


fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set_title("First three PCA dimensions")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()

# Load the Iris dataset
iris = datasets.load_iris()

# Access the features
sepal_length = iris.data[:, 0]  # Sepal Length
sepal_width = iris.data[:, 1]   # Sepal Width
petal_length = iris.data[:, 2]  # Petal Length
petal_width = iris.data[:, 3]   # Petal Width

print(iris)
