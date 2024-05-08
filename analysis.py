# Exploring Iris DataSet using Pandas
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['species'] = iris.target_names[iris.target]
df.to_csv('iris_dataset.csv', index=False)
# Showing the dataset csv
print(df)

# Describe the data set.
df.describe()
