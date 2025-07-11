# Titanic Survival Prediction

This project helps predicting the passenger survival chances on the Titanic using machine learning. By using data science and the historical data present, we can find who had better chances of surviving one of the most infamous shipwrecks in history.

---
 
# Project Overview

The goal of this project is to build a predictive model that answers the question:

"Would this passenger have survived the Titanic disaster?"
or 
"Would this passenger have not survived the titanic wreck?"

We use the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic) 

---

# Technologies Used

- **Python** 
- **Pandas**  
- **NumPy**   
- **Matplotlib & Seaborn** 
- **Scikit-Learn**   
- **Jupyter Notebook** 

---

# ML Workflow

1. **Data Cleaning** – Fix missing values, drop irrelevant columns, convert categories.
2. **Exploratory Data Analysis (EDA)** – Spot patterns & relationships between features and survival.
3. **Feature Engineering** – Extract titles, bin ages, encode categorical variables.
4. **Modeling** – Train classifiers like Logistic Regression, Random Forest, etc.
5. **Evaluation** – Use accuracy, confusion matrix, and cross-validation to select the best model.

---

# Project Structure

titanic-survival-prediction/
├── data/
│ ├── train.csv
│ ├── test.csv
├── notebooks/
│ ├── titanic_analysis.ipynb
├── models/
│ ├── best_model.pkl
├── outputs/
│ ├── prediction.csv
├── README.md


---

## Key Features Considered

- **Pclass** – Passenger class (1st, 2nd, 3rd)
- **Sex** – Gender (surprisingly important!)
- **Age** – Child, adult, elderly
- **SibSp / Parch** – Family onboard
- **Fare** – Ticket price
- **Embarked** – Port of embarkation

---

## Results & Accuracy

After testing multiple algorithms, the best-performing model (e.g., Random Forest) achieved around **79.3% accuracy** on the validation set.  

---

## Future Improvements

- Use ensemble learning (XGBoost / VotingClassifier)
- Tune hyperparameters with GridSearchCV
- Visualize feature importance more interactively
- Deploy the model as a web app (Streamlit or Flask)

---

## Acknowledgements

Thanks to:
- [Kaggle](https://www.kaggle.com/) for the dataset
- Titanic victims and survivors whose stories live on
- The open-source community for endless learning

---

## Author

**Harshit**  
Email- Harshitgulia136@gmail.com  
 [LinkedIn](www.linkedin.com/in/harshit-gulia-b9b909302) | [GitHub](https://github.com/harshit1098)


