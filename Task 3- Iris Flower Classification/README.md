# Task 3 - Iris Flower Classification using Logistic Regression

This is the third project in my *Data Science Internship at CodSoft, where I implemented a **Logistic Regression* model to classify iris flowers into one of three species based on petal and sepal measurements.

---

##  Objective

To build a supervised machine learning model that can classify iris flowers as:
- *Setosa*
- *Versicolor*
- *Virginica*

based on four measurable features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

##  Dataset

- *File Used*: IRIS.csv  
- *Source*: [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) / [Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

---

##  Technologies Used

- *Python*
- *Pandas* – for data manipulation
- *NumPy* – for numerical operations
- *Matplotlib & Seaborn* – for data visualization
- *Scikit-learn* – for model building and evaluation

---

## Model Details

- *Algorithm*: Logistic Regression  
- *Evaluation Metrics*: Accuracy, Classification Report, Confusion Matrix  
- *Model Accuracy: ~**100%* on test data (based on provided example)

---

##  Project Workflow

1. Load and inspect the dataset (df.head(), df.info(), df.describe())
2. Visualize relationships using *Seaborn PairPlot*
3. Encode the categorical target labels using LabelEncoder
4. Split dataset into training and testing sets (80/20)
5. Train a *Logistic Regression* model
6. Predict on the test set
7. Evaluate using accuracy score, classification report, and heatmap of confusion matrix

---

##  Sample Output
Accuracy of the model: 100.00%

Classification Report:
precision recall f1-score support

       0       1.00      1.00      1.00         9
       1       1.00      1.00      1.00        11
       2       1.00      1.00      1.00        10

##  How to Run the Project

1. Download or clone the repository
2. Make sure the file IRIS.csv is in the same folder as your script
3. Run the script with:
python iris_classification.py
Ensure the required libraries are installed:
pip install pandas numpy matplotlib seaborn scikit-learn
##  Demo Video


## ✍ Author
Harshit
📧 Email: harshitgulia136@gmail.com
🔗 LinkedIn:www.linkedin.com/in/harshit-gulia-b9b909302
🐙 GitHub: (http://github.com/harshit1098)

