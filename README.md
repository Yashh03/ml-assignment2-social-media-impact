# Social Media Impact Classifier: ML Assignment 2

Classifying the overall impact of social media use on students with five machine learning models, and demonstrating them in a Streamlit web app.

**Name:** Yash D. Gohil  
**BITS ID:** 2026AJ05448  
**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP  
**GitHub repository:** `https://github.com/Yashh03/ml-assignment2-social-media-impact`  
**Live Streamlit app:** `<paste your Streamlit Community Cloud link here>`

---

## a. Problem statement

Social media can help or harm students depending on how it is used. This project predicts the **overall impact of social media use** on a student (**Beneficial**, **Neutral** or **Negative**) from the student's usage habits, sleep, stress, mental health and academic details.

It is a 3-class classification problem. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same train/test split, tuned with cross-validation, compared on six evaluation metrics, and made available in an interactive Streamlit app that loads the saved models without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Source | Social Media Impact on Life, Kaggle: `https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life` |
| Instances | 4,500 (no duplicate rows) |
| Features | 14 input features. Requirement: at least 12 features and 500 instances |
| Target | `Overall_Impact` with 3 classes |
| Class counts | Beneficial 3,681 (81.80%), Neutral 654 (14.53%), Negative 165 (3.67%). The dataset is strongly imbalanced |

**Features**
- Numeric (8): `Age`, `Daily_Usage_Hours`, `Weekend_Extra_Hours`, `Sleep_Duration_Hours`, `Sleep_Quality_Score`, `Perceived_Stress_Score`, `Mental_Health_Index`, `Academic_Performance_GPA`
- Categorical (5): `Gender`, `Academic_Level`, `Primary_Platform`, `Device_Type`, `Social_Comparison_Frequency`
- True/False (1): `Late_Night_Usage`

**Data preparation**
- Missing values: `Perceived_Stress_Score` (46 rows, 1.02%) and `Academic_Performance_GPA` (85 rows, 1.89%). They are filled with the median inside each model's pipeline, learned from the training data only.
- Numeric features are standardised, categorical features are one-hot encoded (unseen categories are ignored), and the True/False feature is encoded as one binary column. All of this is saved inside each model file, so a saved model accepts the raw CSV columns.
- The data is split 80/20 with stratification (`random_state=42`): 3,600 training rows and 900 test rows (736 Beneficial, 131 Neutral, 33 Negative). The 900 test rows are saved as `test_data.csv`.
- Because the target is imbalanced (about 82% of students are Beneficial), accuracy alone is misleading: a model that always predicts Beneficial gets 81.78% accuracy but an MCC of 0. Precision, Recall and F1 are therefore macro-averaged, AUC is one-vs-rest macro-averaged, and MCC is the multi-class version.

## c. GitHub repository link

`https://github.com/Yashh03/ml-assignment2-social-media-impact`

Files in the repository:

| File | Purpose |
|---|---|
| `2026AJ05448.ipynb` | Complete source code: EDA, preprocessing, training and tuning of the five models, evaluation, saving the models |
| `app.py` | Streamlit app. It loads the saved models and never trains |
| `requirements.txt` | Library versions (the same as on the BITS Virtual Lab) |
| `README.md` | This file |
| `test_data.csv` | The 900 held-out test rows (all 14 features plus `Overall_Impact`) |
| `model/` | The five saved models (`.joblib`, each is a full preprocessing + classifier pipeline) and `metrics_summary.csv` |

## d. Models used

All five models were implemented on the same dataset and the same split. Each is a scikit-learn Pipeline (preprocessing followed by the classifier), tuned with `GridSearchCV` using 5-fold stratified cross-validation on the training set and macro-F1 as the score. The test set was not used for tuning.

Selected settings: Logistic Regression `C=10`; Decision Tree `criterion=entropy`, `max_depth=None`, `min_samples_leaf=1`; KNN `k=11`, distance weighting; Gaussian Naive Bayes `var_smoothing=0.5`; SVM linear kernel, `C=10`, `probability=True`.

### Comparison table (test set, 900 students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9811 | 0.9982 | 0.9537 | 0.9466 | 0.9501 | 0.9388 |
| Decision Tree | 0.9733 | 0.9655 | 0.9158 | 0.9523 | 0.9325 | 0.9145 |
| KNN | 0.9456 | 0.9901 | 0.9327 | 0.8446 | 0.8841 | 0.8152 |
| Naive Bayes | 0.9389 | 0.9880 | 0.8780 | 0.9169 | 0.8950 | 0.8178 |
| Support Vector Machine | **0.9867** | **0.9984** | **0.9609** | **0.9648** | **0.9628** | **0.9570** |

Number of wrong predictions out of 900: SVM 12, Logistic Regression 17, Decision Tree 24, KNN 49, Naive Bayes 55.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best model (accuracy 0.9811, AUC 0.9982, MCC 0.9388) with only 17 errors. The classes are close to linearly separable after standardising the features, so a simple linear model works very well. It is also fast to train and easy to interpret. |
| Decision Tree | Good accuracy (0.9733) and high recall (0.9523), but the lowest AUC of all models (0.9655), because tree leaves give coarse probabilities. The best tree has no depth limit, which suggests some overfitting, and this shows up as lower precision (0.9158): for example, 6 Neutral students are wrongly flagged as Negative. |
| KNN | Accuracy 0.9456 and AUC 0.9901 look good, but its macro recall (0.8446) is the lowest: 35 of 131 Neutral students are predicted as Beneficial, because the neighbours of a borderline student mostly belong to the large Beneficial class. It ranks students well but its final labels are weaker on the minority classes. |
| Naive Bayes | The lowest accuracy (0.9389) and the lowest precision (0.8780). It labels 38 Beneficial students as Neutral. Its independence assumption does not hold here, since usage, sleep, stress and mental health are strongly correlated (r up to -0.85), and one-hot columns are not Gaussian. It needed heavy smoothing (0.5) to work reasonably. |
| Support Vector Machine | Best on all six metrics (accuracy 0.9867, AUC 0.9984, F1 0.9628, MCC 0.9570) with only 12 errors, and balanced results on the rare Negative class (precision and recall both 0.94). It also had the best cross-validated macro-F1 (0.9562). |
| **Overall, winner for the dataset?** | **Support Vector Machine.** Logistic Regression is a very close second (only 5 more mistakes out of 900), so either is a good choice, but SVM leads on every metric, including the macro-F1 and MCC that matter most for an imbalanced dataset. |

## e. Streamlit app

Live app: `<paste your Streamlit Community Cloud link here>`

The app loads the saved models from `model/` and never retrains.

- **Dataset upload (CSV):** upload a test CSV in the sidebar, or tick the bundled `test_data.csv`. A button downloads the sample test file. The uploaded file needs the 14 feature columns and the `Overall_Impact` column. Extra columns and a different column order are fine; a missing column or unknown class label shows an error message.
- **Model selection dropdown:** choose any of the five models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model on the uploaded file.
- **Confusion matrix and classification report:** shown for the chosen model, plus a button to download the predictions as CSV.
- **Model comparison:** a table and a bar chart of all five models on the same file.




