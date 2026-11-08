# Social Media Impact Classifier

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP | **Assignment 2**
**Name:** Yash D. Gohil | **BITS ID:** 2026AJ05448

This project predicts whether a student's social media habits have a Beneficial, Neutral or Negative effect on their life. Five scikit-learn classifiers are built on one dataset, compared on six metrics, and served through a Streamlit web app that only loads the saved models.

- **GitHub repository:** https://github.com/Yashh03/ml-assignment2-social-media-impact
- **Live Streamlit app:** https://social-media-impact-on-life.streamlit.app

---

## a. Problem statement

Whether social media helps or hurts a student depends on how, and how much, it is used. The goal here is to label each student with the **overall impact** of their usage (Beneficial, Neutral or Negative) using information about their screen time, sleep, stress, mental health and studies.

This is a three-class classification task. Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and a Support Vector Machine are trained on identical data, tuned by cross-validation, scored on six metrics, and exposed in an app where a user can upload a test CSV and inspect any model's results without retraining.

## b. Dataset description

| Item | Details |
| --- | --- |
| Source | Kaggle, "Impact of Social Media on Life": https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life |
| Rows | 4,500 students, with no duplicates |
| Input features | 14 (the Student_ID column only identifies a row, so it is dropped). The assignment asks for at least 12 features and 500 rows |
| Target | Overall_Impact, three classes |
| Class distribution | Beneficial 3,681 (81.80%), Neutral 654 (14.53%), Negative 165 (3.67%), so the data is heavily skewed towards one class |

**Input features**

- Numeric (8): Age, Daily_Usage_Hours, Weekend_Extra_Hours, Sleep_Duration_Hours, Sleep_Quality_Score, Perceived_Stress_Score, Mental_Health_Index, Academic_Performance_GPA
- Categorical (5): Gender, Academic_Level, Primary_Platform, Device_Type, Social_Comparison_Frequency
- Boolean (1): Late_Night_Usage

**How the data was prepared**

- **Gaps in the data:** Perceived_Stress_Score has 46 blanks (1.02%) and Academic_Performance_GPA has 85 (1.89%). Each model fills these with the column median, and the median is calculated from training rows only so nothing leaks from the test set.
- **Encoding and scaling:** numeric columns are standardised, categorical columns are one-hot encoded (categories not seen in training are ignored), and Late_Night_Usage becomes a single 0/1 column. These steps live inside each saved model, so a model can be applied directly to a CSV with the original column names.
- **Train/test division:** a stratified 80/20 split with random_state=42 gives 3,600 training rows and 900 test rows (736 Beneficial, 131 Neutral, 33 Negative). The 900 test rows are saved as `test_data.csv`.
- **Choice of metrics:** because about 82% of students are Beneficial, a model that always answers "Beneficial" would reach 81.78% accuracy while having an MCC of 0. Accuracy alone is therefore not enough. Precision, Recall and F1 are macro-averaged, AUC is one-vs-rest with macro averaging, and MCC is the multi-class form.

## c. GitHub repository link

https://github.com/Yashh03/ml-assignment2-social-media-impact

| File / folder | What it contains |
| --- | --- |
| `2026AJ05448.ipynb` | Full notebook: data exploration, preprocessing, tuning and training of all five models, evaluation, and model export |
| `app.py` | The Streamlit app, which loads saved models and never trains |
| `requirements.txt` | Package list (scikit-learn and numpy versions match the BITS Virtual Lab) |
| `README.md` | This file |
| `test_data.csv` | The 900 held-out rows, with all 14 features and Overall_Impact |
| `model/` | Five `.joblib` files (each a complete preprocessing + classifier pipeline) and `metrics_summary.csv` |

## d. Models used

Every model was built on the same split and wrapped in a scikit-learn Pipeline. Hyperparameters were chosen with GridSearchCV (5-fold stratified cross-validation on the training rows, scored by macro-F1). The test rows were kept out of the tuning process entirely.

Chosen settings:

- Logistic Regression: C = 10
- Decision Tree: criterion = entropy, max_depth = None, min_samples_leaf = 1
- KNN: k = 11 with distance weighting
- Gaussian Naive Bayes: var_smoothing = 0.5
- SVM: linear kernel, C = 10, probability = True

### Results on the 900 test students

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9811 | 0.9982 | 0.9537 | 0.9466 | 0.9501 | 0.9388 |
| Decision Tree | 0.9733 | 0.9655 | 0.9158 | 0.9523 | 0.9325 | 0.9145 |
| KNN | 0.9456 | 0.9901 | 0.9327 | 0.8446 | 0.8841 | 0.8152 |
| Naive Bayes | 0.9389 | 0.9880 | 0.8780 | 0.9169 | 0.8950 | 0.8178 |
| Support Vector Machine | 0.9867 | 0.9984 | 0.9609 | 0.9648 | 0.9628 | 0.9570 |

Misclassified students out of 900: SVM 12, Logistic Regression 17, Decision Tree 24, KNN 49, Naive Bayes 55.

### What each model showed

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Came second overall, with 17 mistakes (accuracy 0.9811, AUC 0.9982, MCC 0.9388). After standardisation the three classes are almost separable by a straight boundary, which is why this simple, quick and easy-to-explain model does so well. |
| Decision Tree | Accuracy of 0.9733 and recall of 0.9523 are strong, yet its AUC (0.9655) is the lowest of the five because a tree outputs coarse, step-like probabilities. The tuned tree grew without a depth cap, hinting at some overfitting, and precision suffers (0.9158): 6 Neutral students were mislabelled as Negative. |
| KNN | Accuracy (0.9456) and AUC (0.9901) look healthy, but macro recall of 0.8446 is the weakest of all models. 35 of the 131 Neutral students were labelled Beneficial, since a borderline student's nearest neighbours are mostly from the dominant Beneficial group. It ranks students well but handles the minority classes less reliably. |
| Naive Bayes | Lowest accuracy (0.9389) and lowest precision (0.8780); 38 Beneficial students were called Neutral. Its assumption that features are independent is a poor fit here, because usage, sleep, stress and mental health are strongly related (correlations as high as -0.85), and one-hot columns are not Gaussian. It needed a large smoothing value (0.5) to behave sensibly. |
| Support Vector Machine | Top score on all six metrics (accuracy 0.9867, AUC 0.9984, F1 0.9628, MCC 0.9570) with just 12 errors, and it treats the rare Negative class evenly (precision and recall both 0.94). It also had the highest cross-validated macro-F1 (0.9562) during tuning. |
| **Overall winner for this dataset** | **Support Vector Machine.** Logistic Regression is a close runner-up with only 5 extra errors, so either would be reasonable. SVM is preferred because it leads on every metric, including macro-F1 and MCC, which matter most when classes are this unbalanced. |

## e. Streamlit app

Live app: https://social-media-impact-on-life.streamlit.app

All models are loaded from `model/`; nothing is trained inside the app.

- **CSV upload:** upload a test file from the sidebar, or tick the option to use the bundled `test_data.csv` (there is also a download button for it). The file must have the 14 feature columns plus Overall_Impact.
- **Model dropdown:** pick any of the five classifiers.
- **Metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model, shown against the five-model average.
- **Confusion matrix and classification report:** the matrix can be viewed as counts or row percentages, alongside per-class charts and ROC curves.
- **Compare models:** all five models scored on the same file, with the best model highlighted for each metric.
- **Try a student:** enter the details of one student and see what every model predicts.
- **Data explorer:** class balance and how each feature varies across the classes.
