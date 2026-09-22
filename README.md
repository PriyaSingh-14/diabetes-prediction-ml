# 🩺 Diabetes Risk Prediction using Machine Learning

---

## 1. Problem Statement
Diabetes mellitus is a chronic metabolic disease characterized by elevated levels of blood glucose, which can lead to serious long-term damage to the heart, blood vessels, eyes, kidneys, and nerves over time. Early detection of diabetes is critical for enabling lifestyle interventions and medical treatment to manage or prevent adverse outcomes. 

The goal of this project is to build, evaluate, and optimize a binary supervised machine learning classification model to predict whether a patient has diabetes based on diagnostic physiological metrics. By leveraging clinical indicators such as Glucose level, BMI, Insulin, and Age, the system aims to provide an accurate early-warning prediction mechanism.

---

## 2. Dataset Description
The model is trained on the **PIMA Indians Diabetes Dataset**, sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. All subjects in this dataset are female patients of Pima Indian heritage aged at least 21 years old.

* **Total Records:** 768 patient instances
* **Total Features:** 8 numerical predictive features + 1 binary target feature (`Outcome`)

| Feature Name | Description | Data Type | Range / Note |
| :--- | :--- | :--- | :--- |
| **Pregnancies** | Number of times pregnant | `int64` | $0 - 17$ |
| **Glucose** | Plasma glucose concentration (2 hours in an oral glucose tolerance test) | `int64` | $0 - 199$ mg/dL (0 = missing value) |
| **BloodPressure** | Diastolic blood pressure | `int64` | $0 - 122$ mm Hg (0 = missing value) |
| **SkinThickness** | Triceps skin fold thickness | `int64` | $0 - 99$ mm (0 = missing value) |
| **Insulin** | 2-Hour serum insulin | `int64` | $0 - 846$ mu U/ml (0 = missing value) |
| **BMI** | Body mass index (weight in kg / (height in m)²) | `float64` | $0 - 67.10$ (0 = missing value) |
| **DiabetesPedigreeFunction** | Diabetes pedigree score (genetic/family history score) | `float64` | $0.078 - 2.42$ |
| **Age** | Patient age in years | `int64` | $21 - 81$ years |
| **Outcome** | Target Variable ($0$ = Non-Diabetic, $1$ = Diabetic) | `int64` | $500$ Non-Diabetic ($65.1\%$), $268$ Diabetic ($34.9\%$) |

---

## 3. Data Preprocessing
1. **Handling Biologically Impossible Zero Values:**
   Initial statistical profiling (`df.describe()`) revealed minimum values of `0` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Because a physiological measurement of `0` in these fields is medically impossible, these were flagged as missing data encoded as `0`.
   * Columns Imputed: `['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']`
   * Strategy: Replaced `0` values with `np.nan` and filled them using column-wise **Median Imputation** (`fillna(df[col].median())`) to avoid skewing distributions with outliers.
2. **Train-Test Stratified Splitting:**
   * Split ratio: **80% Training ($614$ samples)** and **20% Testing ($154$ samples)** using `random_state=42`.
   * Stratification (`stratify=y`) was applied to ensure the $65/35$ target class balance was maintained proportionally across both training and testing sets.
3. **Feature Scaling:**
   * Applied `StandardScaler` inside a scikit-learn `Pipeline` for Logistic Regression to standardize features to zero mean and unit variance.

---

## 4. Exploratory Data Analysis (EDA)
During exploratory analysis:
* **Class Imbalance:** Identified that $65.1\%$ ($500$ cases) were Non-Diabetic ($0$) and $34.9\%$ ($268$ cases) were Diabetic ($1$).
* **Distribution & Missingness:** Inspected summary statistics, confirming no duplicate rows (`df.duplicated().sum() == 0`) and zero explicit missing nulls prior to $0$-value replacement.
* **Feature Relationship:** Visualized class distributions using Seaborn count plots and assessed feature contribution ranks via tree-based feature importance algorithms.

---

## 5. Model Building
Two distinct classification paradigms were developed, trained, and compared:

1. **Logistic Regression (Linear Baseline Model):**
   * Combined with `StandardScaler` inside a scikit-learn `Pipeline` to normalize numerical scale differences across features.
   * Parameter setting: `random_state=42`.
2. **Random Forest Classifier (Ensemble Model):**
   * Built an ensemble of decision trees (`n_estimators=100`, `random_state=42`).
3. **Hyperparameter Tuning (`GridSearchCV`):**
   * Applied 5-fold cross-validation (`cv=5`) targeting `f1` score optimization over the parameter grid:
     ```python
     param_grid = {
         'n_estimators': [100, 200],
         'max_depth': [None, 5, 10],
         'min_samples_split': [2, 5]
     }
     ```
   * Best parameters selected: `min_samples_split=2`, `n_estimators=100`.

---

## 6. Model Evaluation
The models were evaluated on the held-out test dataset ($154$ unseen patient samples) using Accuracy, F1-Score, and Confusion Matrix analysis.

### Performance Summary

| Metric | Logistic Regression | Random Forest (Baseline) | Tuned Random Forest (`GridSearchCV`) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $70.78\%$ | $77.92\%$ | **$77.92\%$** |
| **F1-Score** | $0.5455$ | $0.6531$ | **$0.6531$** |

### Test Set Confusion Matrix Breakdown ($N = 154$)

| Confusion Matrix Quadrant | Logistic Regression | Random Forest Classifier |
| :--- | :---: | :---: |
| **True Negatives (TN)** (Actual: 0, Pred: 0) | $82$ | **$88$** |
| **False Positives (FP)** (Actual: 0, Pred: 1) | $18$ | **$12$** |
| **False Negatives (FN)** (Actual: 1, Pred: 0) | $27$ | **$22$** |
| **True Positives (TP)** (Actual: 1, Pred: 1) | $27$ | **$32$** |

---

## 7. Interpretation
* **Model Comparison:** Random Forest outperformed Logistic Regression by **$+7.14\%$** in accuracy ($77.92\%$ vs $70.78\%$) and significantly boosted the F1-Score from $0.5455$ to $0.6531$.
* **Medical Risk Reduction:** In diagnostic AI, **False Negatives** represent the most critical risk (failing to diagnose a diabetic patient). Random Forest reduced False Negatives from $27$ down to $22$ while simultaneously reducing False Alarms (False Positives) from $18$ to $12$.
* **Feature Importance Ranking:**
  1. **Glucose:** Highest predictive influence ($>27\%$ total feature weight).
  2. **BMI:** Second most important risk indicator (~16\%).
  3. **DiabetesPedigreeFunction & Age:** Moderate secondary predictive impact (~11\% - 12\%).
  4. **Insulin, BloodPressure, Pregnancies, & SkinThickness:** Lower individual relative weights (<10\%).

---

## 8. Conclusion & Limitations

### Conclusion
The project successfully established a functional machine learning pipeline capable of detecting diabetes with **$77.92\%$ accuracy**. Random Forest proved superior to linear models due to its ability to capture non-linear relationships and interactions between clinical parameters like Glucose, BMI, and Age.

### Limitations
1. **Demographic Specificity:** The dataset consists exclusively of female subjects of Pima Indian heritage aged $\ge 21$, limiting direct generalizability to broader demographic populations or male patients.
2. **Dataset Volume:** The dataset contains $768$ total samples, which is relatively small for training complex deep learning architectures without risk of overfitting.
3. **High Zero Imputation Volume:** Columns such as `Insulin` and `SkinThickness` contained significant proportions of missing `0` values requiring median replacement.

---

## 9. Future Scope
* **Interactive UI Deployment:** Build and host a interactive web user interface using **Streamlit** or **Gradio** to allow clinical practitioners to adjust patient sliders and receive real-time probability estimates.
* **Advanced Ensembles & Imbalance Handling:** Explore gradient boosting algorithms (**XGBoost**, **LightGBM**) and synthetic oversampling techniques (**SMOTE**) to further reduce False Negatives.
* **REST API Endpoint:** Wrap the saved model object (`joblib` `.pkl`) inside a **FastAPI** backend to serve low-latency prediction endpoints for third-party healthcare systems.

---

## 10. Technology Stack
* **Language:** Python 3.14
* **Environment:** JupyterLab / Jupyter Notebook
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `Pipeline`, `StandardScaler`)
* **Data Visualization:** Matplotlib, Seaborn
* **Model Persistence:** Joblib

---

## 11. Project Structure
```text
diabetes-prediction-ml/
├── diabetes_prediction.ipynb     # Main Jupyter Notebook containing the full pipeline
├── diabetes.csv                  # PIMA Indians Diabetes Dataset
├── requirements.txt              # Dependency packages list for environment setup
└── README.md                     # Comprehensive project documentation
```

## 12. Author
* **Authored by:** Priya Bhadoriya
* **Repository:** https://github.com/PriyaSingh-14/diabetes-prediction-ml
