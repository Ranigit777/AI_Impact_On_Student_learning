# Model Evaluation & Selection Report

**Project Title**: AI Impact on Student Learning – Burnout Risk Prediction
**Phase**: Phase 2 - Model Training, Evaluation & Selection

## 1. Model Comparison Summary Table

| Model                  |   Accuracy |   Precision (Weighted) |   Recall (Weighted) |   F1 Score (Weighted) |   5-Fold CV F1 (Mean) |   5-Fold CV F1 (Std) |   Training Time (s) |
|:-----------------------|-----------:|-----------------------:|--------------------:|----------------------:|----------------------:|---------------------:|--------------------:|
| Logistic Regression    |   0.519097 |               0.531111 |            0.519097 |              0.514481 |              0.514931 |           0.00260129 |             1.72296 |
| Support Vector Machine |   0.517064 |               0.527847 |            0.517064 |              0.512224 |              0.511432 |           0.00449421 |             2.90075 |
| Gradient Boosting      |   0.513534 |               0.523931 |            0.513534 |              0.508817 |              0.510856 |           0.00506807 |             6.42985 |
| XGBoost                |   0.511608 |               0.522198 |            0.511608 |              0.507624 |              0.504632 |           0.00361856 |             3.63535 |
| Random Forest          |   0.495774 |               0.501253 |            0.495774 |              0.492864 |              0.491878 |           0.00430871 |             4.99649 |
| K-Nearest Neighbors    |   0.434578 |               0.434887 |            0.434578 |              0.433344 |              0.429134 |           0.00811396 |             2.02353 |
| Decision Tree          |   0.422275 |               0.422496 |            0.422275 |              0.422371 |              0.423843 |           0.00342214 |             4.70078 |

## 2. Best Performing Model: `Logistic Regression`

- **Baseline F1 Score**: `0.5145`
- **Tuned F1 Score**: `0.5146`
- **Best Parameters**: `{'C': 10.0}`

### Classification Report for Best Model

```text
              precision    recall  f1-score   support

        High       0.61      0.35      0.45      2021
         Low       0.54      0.50      0.52      3228
      Medium       0.49      0.61      0.54      4098

    accuracy                           0.52      9347
   macro avg       0.54      0.49      0.50      9347
weighted avg       0.53      0.52      0.51      9347
```

## 3. Top 15 Feature Importances

| Feature                                    |   Importance |
|:-------------------------------------------|-------------:|
| Weekly_GenAI_Hours                         |    0.384271  |
| Year_of_Study_Graduate                     |    0.359524  |
| Year_of_Study_Freshman                     |    0.315096  |
| Year_of_Study_Sophomore                    |    0.150132  |
| Institutional_Policy_Actively_Encouraged   |    0.112989  |
| Perceived_AI_Dependency                    |    0.107026  |
| Institutional_Policy_Allowed_With_Citation |    0.0978497 |
| Institutional_Policy_Strict_Ban            |    0.0771806 |
| Traditional_Study_Hours                    |    0.075189  |
| Year_of_Study_Senior                       |    0.0676683 |
| Pre_Semester_GPA                           |    0.0591407 |
| Prompt_Engineering_Skill_Advanced          |    0.0582528 |
| Post_Semester_GPA                          |    0.0563856 |
| Primary_Use_Case_Ideation                  |    0.0544218 |
| Year_of_Study_Junior                       |    0.0516829 |

## 4. Model Artifact Saved

- Trained Pipeline artifact path: `models/burnout_prediction_model.pkl`
