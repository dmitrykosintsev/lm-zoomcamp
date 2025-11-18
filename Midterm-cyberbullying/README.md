# lm-zoomcamp
## Problem statement
As a teacher, I was looking into current problems that could be analysed and solved using machine learning. I came across a report from UNDP Mongolia about cyberbullying. Though I could not find any available raw data for the report, it was the most up-to-date information about any school-related issues in Mongolia. I decided to generate a synthetic dataset from the report in order to create models that could predict cyberbullying in the context of Mongolia.

My goal is to achieve such accuracy that will allow teachers and parents to use the model in order to identify potential cases of cyberbullying.

## Data & EDA insights
There were many missing values due to the fact some variables only had values when experienced_cyberbullying was True. For details, check the notebook.

## Model choice
I trained and tuned the following models:
- Linear Regression: best model with C=0.01 has AUC=0.5968 and F1 score=0.631
- Decision tree: best model with max_depth=2, min_samples_leaf=500 has AUC=0.5867 and F1 score=0.619
- Random Forest: best model with n_estimators=100, max_depth=5, min_samples_leaf=50 has AUC=0.5933 and F1 score=0.625
- GBoost: best model with eta=0.01, max_depth=4, min_child_weight=1 has AUC=0.5821 and F1 score=0.626

## Deployment
In progress