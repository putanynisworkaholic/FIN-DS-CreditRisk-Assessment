# Credit Risk Assessment Project

Welcome to my Credit Risk Assessment project! I'm Putanyn Manee, and in this repository, I explore various machine learning models to predict loan defaults. This journey involves data preprocessing, model training, hyperparameter tuning, and performance evaluation.

## Project Overview

This project aims to build reliable models to predict whether a borrower will default on their loan. Accurate predictions can help financial institutions mitigate risks and make informed lending decisions.

## Introduction

In this project, I explore a dataset containing information about borrowers' financial status, loan details, and credit history. The goal is to develop models that can accurately predict loan defaults. The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/urvishvekariya/credit-risk-assessment).

## Data Exploration and Preprocessing

To start, I dive into the data, handle missing values, transform skewed features, encode categorical variables, and scale the features. This ensures that the data is clean and ready for modeling.

## Model Training and Evaluation

I train several machine learning models, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Before training, I'll use Optuna to fine-tune the hyperparameters for each model. This will help me find the best settings for optimal performance. I’ll evaluate each model using accuracy, F1 score, and confusion matrix to see how well they perform on the test data.

## Hyperparameter Tuning

Using Optuna, I fine-tune the hyperparameters for each model to achieve optimal performance. Here are the best parameters found:

| Classifier           | Parameters                                                           |
|----------------------|----------------------------------------------------------------------|
| **LightGBM**         | lgb_n_estimators: 268<br>lgb_max_depth: 16<br>lgb_learning_rate: 0.0844 |
| **Naive Bayes**      | No hyperparameters to tune.                                          |
| **XGBoost**          | xgb_n_estimators: 254<br>xgb_max_depth: 6<br>xgb_learning_rate: 0.0955 |
| **SVM**              | svc_C: 86.896                                                        |
| **Logistic Regression** | lr_C: 70.837                                                      |
| **CatBoost**         | cb_n_estimators: 296<br>cb_learning_rate: 0.0599<br>cb_depth: 8      |
| **Random Forest**    | rf_n_estimators: 27<br>rf_max_depth: 28                              |
| **KNN**              | n_neighbors: 5                                                       |

## Results Summary

After evaluating the models, LightGBM, XGBoost, and CatBoost emerged as the top performers, each with an impressive AUC of 0.94. These models are highly effective at predicting loan defaults.

## Conclusion

Wow, what a journey! 🎉 I, Putanyn Manee, have taken a deep dive into the world of credit risk assessment, and it's been quite an adventure. Here’s a quick recap of what I've accomplished:

1. Explored and Preprocessed the Data
2. Trained and Evaluated Several Models
3. Tuned Hyperparameters to Optimize Performance
4. Identified LightGBM, XGBoost, and CatBoost as Top Performers

## Future Work

But wait, there's more! 🚀 For future work, I plan to explore ensemble models. By combining the strengths of multiple models, I can potentially achieve even better performance. Stay tuned for more exciting developments!


## Usage

To use this project, simply clone the repository, install the dependencies, and run the Jupyter Notebook:

```bash
git clone https://github.com/PutanynManee/credit-risk-assessment.git
cd credit-risk-assessment
jupyter notebook
```

Open the `credit_risk_assesment.ipynb` notebook to see the complete analysis and results.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details. The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/urvishvekariya/credit-risk-assessment) and is available under the terms provided by Kaggle.
```
