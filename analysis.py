# Exploring Iris DataSet using Pandas

from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['species'] = iris.target_names[iris.target]
df.to_csv('iris_dataset.csv', index=False)