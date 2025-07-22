# Machine Learning: Decision Tree Classifier Projects

This repository contains a Jupyter Notebook (`Decision_Tree_practical.ipynb`) that demonstrates the implementation of the Decision Tree algorithm for classification tasks using Python's Scikit-learn library. The notebook covers two distinct examples to illustrate the model's application and evaluation.

## 🌳 Core Concepts of Decision Trees

A **Decision Tree** is a supervised machine learning algorithm that is used for both classification and regression tasks. It works by creating a tree-like model of decisions. Each internal node represents a "test" on an attribute (e.g., whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (a decision taken after computing all attributes).

The notebook implicitly uses key criteria for creating splits in the tree:
* **Gini Impurity:** A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The algorithm aims to find splits that minimize Gini impurity.
* **Information Gain:** The reduction in entropy or surprise by splitting a dataset based on a given attribute. A higher information gain indicates a more effective split.

---

## 🛍️ Project 1: Social Network Ads Classification

This project builds a Decision Tree classifier to predict user purchasing behavior.

* **Goal:** To predict whether a user will purchase a product based on their **Age** and **Estimated Salary**.
* **Dataset:** `Social_Network_Ads.csv`
* **Steps Covered:**
    1.  **Data Preprocessing:** The dataset is loaded, and features (X) and the target variable (y) are separated.
    2.  **Train-Test Split:** The data is divided into training and testing sets to evaluate the model on unseen data.
    3.  **Feature Scaling:** `StandardScaler` is applied to normalize the feature values, ensuring that no single feature dominates the model's learning process.
    4.  **Model Training:** A `DecisionTreeClassifier` is initialized (using the `entropy` criterion for information gain) and trained on the scaled training data.
    5.  **Evaluation:** The model's performance is assessed using a **Confusion Matrix** and **Accuracy Score**.
    6.  **Visualization:** The decision boundaries for the training and test sets are plotted to provide a visual representation of how the classifier separates the data points.

### Code: Model Implementation
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree Classification model on the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and checking accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

---

## 🌸 Project 2: Iris Flower Classification & Tree Visualization

This project uses the classic Iris dataset to demonstrate how to visualize the internal structure of a trained decision tree.

* **Goal:** To classify iris flowers into one of three species (`setosa`, `versicolor`, `virginica`) based on sepal and petal measurements.
* **Dataset:** The Iris dataset, loaded directly from Scikit-learn.
* **Key Feature: Tree Visualization:** The main focus of this section is to create a visual plot of the decision tree itself. This helps in understanding how the model makes its predictions by showing the decision rules at each node.

### Code: Visualizing the Decision Tree
```python
from sklearn import tree
import matplotlib.pyplot as plt

# Assume 'classifier' is the trained DecisionTreeClassifier from the Iris dataset
plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled=True)
plt.show()
```
This visualization displays the nodes, the Gini impurity at each node, the number of samples, and the predicted class, offering deep insight into the model's logic.

---

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    ```
2.  **Navigate to the directory:**
    ```bash
    cd <your-repository-name>
    ```
3.  **Install dependencies:** Ensure you have the necessary libraries installed.
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```
4.  **Launch Jupyter Notebook and open `Decision_Tree_practical.ipynb`** to explore the code and results.
