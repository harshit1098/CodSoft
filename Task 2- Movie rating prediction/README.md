# Movie Rating Prediction

This project explores how we can predict movie ratings based on data such as genre, cast, budget, release details, and more. By analyzing patterns in historical film data, we aim to forecast the likely audience or critic rating of a new movie before its release.

---

## Objective

The goal is to build a regression model that can predict a movie’s rating (e.g., IMDb score) using relevant features from the dataset. This can be useful for production houses, investors, and streaming platforms to evaluate a film’s potential success.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- Jupyter Notebook

---

## Project Structure



movie-rating-prediction/
├── data/
│ ├── movies.csv
│ ├── cleaned_data.csv
├── notebooks/
│ ├── rating_prediction.ipynb
├── models/
│ ├── best_model.pkl
├── outputs/
│ ├── final_predictions.csv
├── README.md


---

## Features Considered

- Genre (one-hot encoded)
- Director popularity (based on average previous ratings)
- Actor/Actress score (average past movie performance)
- Budget
- Runtime
- Release month
- Language
- Country
- Production company

---

## ML Pipeline

1. Data Preprocessing – Handling null values, encoding categorical data, dropping irrelevant columns.
2. Exploratory Data Analysis (EDA) – Visualizing trends and understanding feature impact.
3. Feature Engineering – Creating new features such as director/cast popularity score.
4. Model Training – Using algorithms like Linear Regression, Random Forest, and Gradient Boosting.
5. Model Evaluation – R² score, RMSE, MAE, and cross-validation for performance measurement.

---

## Results

The final model achieved an R² score of X.XX and an RMSE of Y.YY on the validation dataset.  
*(Replace X.XX and Y.YY with actual performance metrics.)*

---

## Future Improvements

- Add textual analysis using plot summaries with NLP
- Integrate user review sentiment analysis
- Include real-time scraping from IMDb or TMDb
- Build a frontend dashboard using Streamlit or Flask

---

## Acknowledgements

- IMDb / TMDb for providing open movie data  
- Open-source Python libraries for enabling rapid prototyping  
- The data science community for learning resources and support

---

## Author

**Harshit Gulia**  
Email: harshitgulia136@gmail.com  
LinkedIn:(https://www.linkedin.com/in/harshit-gulia-b9b909302)  
GitHub: (https://github.com/harshit1098)

---

## Demo Video 

https://www.linkedin.com/posts/harshit-gulia-b9b909302_python-datascience-mlprojects-activity-7349774639503491073-_W4o?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE1gj2cBTTz7nERtkU5M40KuAvzhXcfSUuA
