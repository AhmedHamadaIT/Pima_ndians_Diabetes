# Diabetes Prediction Project

This project focuses on predicting diabetes using the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The dataset contains medical information about patients, and the goal is to predict whether a patient has diabetes (Outcome = 1) or not (Outcome = 0). The project involves data preprocessing, exploratory data analysis (EDA), outlier handling, model training, evaluation, and comparison.


## Dataset Description

The dataset used in this project is the Pima Indians Diabetes Dataset, which contains the following features:
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test).
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / (height in m)^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function (a genetic risk score).
- **Age**: Age in years.
- **Outcome**: Target variable (0 = no diabetes, 1 = diabetes).

## Steps in the Project

### 1. Import Libraries

The necessary Python libraries are imported for data manipulation, visualization, and machine learning:
- `pandas`, `numpy` for data manipulation.
- `matplotlib`, `seaborn` for data visualization.
- `scikit-learn` for machine learning models and evaluation metrics.

### 2. Load and Explore the Dataset

The dataset is loaded using pandas, and basic exploratory data analysis (EDA) is performed:
- Check the first few rows of the dataset.
- Get information about the dataset (data types, missing values).
- Generate descriptive statistics.
- Check for missing values and correlations.

### 3. Exploratory Data Analysis (EDA)

Visualizations are created to understand the distribution of features and relationships between variables:
- Box plots to identify outliers.
- Histograms to visualize feature distributions.
- Scatter plots and pair plots to explore relationships.
- Heatmap to analyze correlations between features.

### 4. Handle Outliers

Outliers are handled using the Interquartile Range (IQR) method and Isolation Forest:
- Outliers are capped using IQR.
- Isolation Forest is used to detect and remove outliers.

### 5. Prepare Data for Modeling

The dataset is split into features (X) and target (y). Features are scaled using StandardScaler, and the data is split into training and testing sets.

### 6. Train and Evaluate Models

Six machine learning models are trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

For each model, the following steps are performed:
- Train the model on the training data.
- Make predictions on the test data.
- Evaluate the model using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- Generate visualizations like Confusion Matrix, ROC Curve, and Precision-Recall Curve.

### 7. Compare Models

The performance of all models is compared using a summary table and bar plots for metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

## Results and Model Comparison

The Model Comparison Table provides a summary of the performance metrics for each machine learning model:

| Model                 | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 0.7532   | 0.6545    | 0.6545 | 0.6545   | 0.8145  |
| Decision Tree         | 0.7273   | 0.6102    | 0.6545 | 0.6316   | 0.7111  |
| Random Forest         | 0.7468   | 0.6379    | 0.6727 | 0.6549   | 0.8122  |
| SVM                   | 0.7208   | 0.6154    | 0.5818 | 0.5981   | 0.8054  |
| KNN                   | 0.7273   | 0.6383    | 0.5455 | 0.5882   | 0.7727  |
| Gradient Boosting     | 0.7597   | 0.6500    | 0.7091 | 0.6783   | 0.8006  |

### Key Insights
- **Best Overall Model**: Gradient Boosting performs the best in terms of Accuracy, Recall, and F1-Score.
- **Best AUC-ROC**: Logistic Regression has the highest AUC-ROC score, indicating it is the best at distinguishing between the two classes.
- **Precision**: Random Forest has the highest precision, meaning it has the fewest false positives.
- **Recall**: Gradient Boosting has the highest recall, meaning it has the fewest false negatives.

### Recommendations
- Use Gradient Boosting if the goal is to maximize overall accuracy and balance between precision and recall.
- Use Logistic Regression if the goal is to maximize the ability to distinguish between classes (AUC-ROC).
- Use Random Forest if minimizing false positives (high precision) is critical.

## How to Run the Code
1. Ensure you have the required libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
2. Download the dataset (`diabetes.csv`) and place it in the working directory.
3. Run the code cells in a Jupyter Notebook or Google Colab.

## Future Work
- Experiment with hyperparameter tuning for better model performance.
- Use advanced techniques like cross-validation and ensemble methods.
- Deploy the best model as a web application using Flask or Streamlit.

## Contact Us 
- Mail : (ahmed1hamada1shabaan@gmail.com)
- kaggle : (https://www.kaggle.com/ahmedxhamada)
- Linkedin : (www.linkedin.com/in/ahmed-hamadaai)