# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : SAM NIKITH SINGH R

*INTERN ID* : CT04DF1181

*DOMAIN NAME* : PYTHON PROGRAMMIMG

*DURATION* : 4 WEEKS

*MENTOR NAME* : NEELA SANTHOSH


**Titanic Survival Prediction – Random Forest Classifier**

 *Project Overview*

This project showcases a **machine learning classification model** that predicts passenger survival on the Titanic using the classic **Titanic dataset**. It uses the `RandomForestClassifier` from the **scikit-learn** library and includes all key steps in the machine learning pipeline: data loading, preprocessing, model training, prediction, evaluation, and visualization.

The Titanic dataset is a widely used beginner-level dataset in data science and machine learning due to its simplicity, real-world context, and structured nature. The goal of this project is to demonstrate how we can build an effective prediction model using a relatively small dataset and a powerful yet interpretable algorithm like Random Forest.

This project is suitable for students, beginners, and intermediate-level practitioners looking to understand supervised learning workflows in Python.

 *Dataset Description*

The dataset used is the **Titanic dataset**, preloaded from the `seaborn` library. It contains detailed passenger information such as:

* Class (`pclass`)
* Sex (`sex`)
* Age (`age`)
* Number of siblings/spouses aboard (`sibsp`)
* Number of parents/children aboard (`parch`)
* Fare paid (`fare`)
* Port of embarkation (`embarked`)
* Whether the passenger was alone (`alone`)
* Survival status (`survived`) — the target variable

After loading, several non-informative or redundant columns are dropped: `deck`, `embark_town`, `alive`, `who`, and `class`.

 *Data Preprocessing*

The dataset is cleaned to handle missing values and categorical data:

* Missing values in critical columns like `age` and `embarked` are dropped.
* The `sex` and `embarked` categorical variables are encoded numerically using `LabelEncoder`.
* The `alone` boolean column is converted to integer values for compatibility with scikit-learn models.

This preprocessing ensures the data is suitable for machine learning algorithms, which generally require numerical input.

 *Feature and Target Selection*

The input features (`X`) include 8 attributes believed to influence survival probability:

* `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, and `alone`.

The target variable (`y`) is the `survived` column, which is binary:

* 1 indicates survival
* 0 indicates non-survival
  
 *Model Training* - *most important part of our project*

The dataset is split into training and testing sets using an 80-20 split via `train_test_split`. The **Random Forest Classifier** is initialized with 100 estimators (trees) and a fixed random seed for reproducibility. The model is then trained on the training data using the `.fit()` method.

Random Forest is chosen for its robustness, ability to handle both numerical and categorical variables (after encoding), and its capability to reduce overfitting through ensemble learning.

 *Model Evaluation*

After training, predictions are made on the test data (`X_test`). The performance of the model is evaluated using:

* **Accuracy Score**: Measures the percentage of correct predictions.
* **Confusion Matrix**: Shows the breakdown of true positives, true negatives, false positives, and false negatives.
* **Classification Report**: Provides precision, recall, F1-score, and support for both classes.

These metrics give a comprehensive view of model performance, beyond just accuracy.
 *Feature Importance Visualization*

To interpret the model, feature importances are extracted from the trained classifier. These values represent how important each feature was in determining the survival outcome.

A **Seaborn barplot** is generated to visualize the relative importance of each feature. This helps identify which factors (like `sex`, `fare`, or `age`) had the most influence on survival prediction, adding transparency to the model’s decision-making process.

 *How to Use*

1. Open the script in **Google Colab** or a local Jupyter Notebook.
2. Ensure the necessary libraries (`seaborn`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`) are installed.
3. Run all cells to:

   * Load and explore the dataset
   * Preprocess and encode data
   * Train the model and evaluate it
   * Visualize feature importance
  
**OUTPUT**

