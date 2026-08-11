# Exploratory Data Analysis & Data Cleaning Summary Report

**Project Title**: AI Impact on Student Learning – Burnout Risk Prediction and Academic Performance Analysis  
**Phase**: Phase 1 - Data Engineering, Cleaning, EDA & Feature Engineering  
**Dataset**: `ai_student_learning_raw.csv` (50,001 Records)  
**Processed Dataset**: `cleaned_student_data.csv`

---

## 1. Executive Summary

This report presents Phase 1 analysis of the **AI Impact on Student Learning** dataset, examining how Artificial Intelligence tool usage (GenAI hours, prompt engineering skills, tool diversity, perceived dependency) interacts with academic performance (Pre/Post Semester GPA), study habits (traditional study hours), and exam anxiety to influence student burnout risk.

---

## 2. Dataset Overview

### Raw Dataset Characteristics
- **Total Records**: 50,001
- **Total Features**: 16
- **Feature Categorization**:
  - **Identifiers**: `Student_ID`
  - **Academic Performance**: `Pre_Semester_GPA`, `Post_Semester_GPA`, `Year_of_Study`, `Major_Category`
  - **AI Tool Habits**: `Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Paid_Subscription`, `Perceived_AI_Dependency`
  - **Institutional Context**: `Institutional_Policy`
  - **Wellbeing & Retention**: `Traditional_Study_Hours`, `Anxiety_Level_During_Exams`, `Skill_Retention_Score`
  - **Target Variable**: `Burnout_Risk_Level` (`Low`, `Medium`, `High`)

---

## 3. Data Cleaning Pipeline

The data cleaning pipeline executed the following steps:
1. **Missing Value Imputation**:
   - Evaluated column-wise null counts.
   - Applied median imputation for numerical features and mode imputation for categorical features.
2. **Duplicate Row Removal**:
   - Identified and removed duplicate records to prevent sampling bias.
3. **Data Type Correction**:
   - Converted categorical strings (`Major_Category`, `Year_of_Study`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Paid_Subscription`, `Institutional_Policy`, `Burnout_Risk_Level`) into categorical data types.
   - Enforced float/int numerical dtypes across GPA, hours, and score columns.
4. **Outlier Detection & Removal**:
   - Utilized the Interquartile Range (IQR) method with a boundary factor of 1.5 across key metrics (`Weekly_GenAI_Hours`, `Traditional_Study_Hours`, `Pre_Semester_GPA`, `Post_Semester_GPA`, `Skill_Retention_Score`).

---

## 4. Feature Engineering

Four domain-specific engineered features were created to enhance predictive power for upcoming ML model development:

1. **GPA Improvement**:
   $$\text{GPA\_Improvement} = \text{Post\_Semester\_GPA} - \text{Pre\_Semester\_GPA}$$
   *Captures academic trajectory over the semester.*

2. **AI Efficiency**:
   $$\text{AI\_Efficiency} = \frac{\text{GPA\_Improvement}}{\text{Weekly\_GenAI\_Hours} + 1.0}$$
   *Measures academic GPA return per hour of GenAI usage.*

3. **Study Balance**:
   $$\text{Study\_Balance} = \frac{\text{Traditional\_Study\_Hours}}{\text{Weekly\_GenAI\_Hours} + 1.0}$$
   *Quantifies the ratio between traditional self-study and AI reliance.*

4. **Dependency Index**:
   $$\text{Dependency\_Index} = \frac{\text{Perceived\_AI\_Dependency} \times \text{Weekly\_GenAI\_Hours}}{\text{Traditional\_Study\_Hours} + 1.0}$$
   *Composite metric indicating high AI dependency relative to traditional study hours.*

---

## 5. Key EDA Findings

1. **Burnout Risk vs. AI Usage**:
   - Students spending $> 20$ weekly hours on GenAI while maintaining low traditional study hours ($< 8$ hours) demonstrate significantly higher proportions of **High Burnout Risk**.
2. **Exam Anxiety & Burnout**:
   - A strong positive correlation exists between `Anxiety_Level_During_Exams` and `Burnout_Risk_Level`. Students reporting anxiety levels $\ge 7$ overwhelmingly fall into `Medium` or `High` burnout categories.
3. **Prompt Engineering Skill & Retention**:
   - Higher `Prompt_Engineering_Skill` (Advanced) correlates with better `Skill_Retention_Score` even when GenAI usage is moderate to high, indicating that proficient prompt usage supports learning rather than substituting for comprehension.
4. **GPA Dynamics**:
   - Moderate GenAI usage ($5 - 12$ hours/week) is associated with slight positive `GPA_Improvement`, whereas extreme usage ($> 25$ hours/week) correlates with diminishing or negative GPA returns.

---

## 6. Exported Visualizations (`reports/figures/`)

- `correlation_heatmap.png`: Matrix of feature correlations.
- `burnout_risk_distribution.png`: Target variable class distribution.
- `ai_usage_distribution.png`: GenAI usage KDE and histogram.
- `gpa_distribution.png`: Pre vs. Post GPA comparison.
- `study_hours_distribution.png`: Scatter plot of Traditional vs. AI study hours.
- `anxiety_distribution.png`: Exam anxiety boxplots across burnout tiers.
- `histograms.png`: Distribution grid of numerical attributes.
- `boxplots.png`: Outlier visualization boxplots.
- `pairplots.png`: Multi-feature pairwise scatter matrix.
- `countplots.png`: Categorical feature breakdowns.
