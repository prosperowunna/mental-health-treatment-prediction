# Predicting Mental Health Treatment-Seeking in Tech Workplaces

A machine learning project that predicts whether a tech worker has sought treatment for a mental health condition, based on their background and workplace environment. The project compares three classification models and identifies the workplace factors most linked to people getting help.

## Results

Evaluated on a held-out test set of 252 responses (20% of the data):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | **0.710** | 0.724 | 0.695 | 0.709 |
| Random Forest | 0.706 | 0.721 | 0.688 | 0.704 |
| Decision Tree (tuned: max depth 4, min leaf 10) | 0.659 | 0.667 | 0.656 | 0.661 |

**Key findings:**
- **Logistic Regression performed best, at 71.0% accuracy**, slightly ahead of Random Forest. The simple linear model was enough for this mostly categorical survey data.
- **Family history of mental illness was by far the strongest predictor** of seeking treatment, followed by **age**.
- **Workplace support matters:** whether the employer offers **mental health care options** and **mental health benefits** were among the top predictors, suggesting that employer-provided resources are linked to people getting help.

These are associations, not causes. For example, people who seek treatment may also be more aware of their employer's benefits.

## Data

[OSMI Mental Health in Tech Survey (2014)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey): 1,259 survey responses (637 sought treatment, 622 did not) from tech workers, covering demographics, employer mental health benefits, and workplace attitudes. The data is not included in this repo; the notebook downloads it automatically with `kagglehub`.

## Approach

1. **Data cleaning:** removed impossible ages (for example, negative values), grouped dozens of free-text gender spellings into three categories, and filled missing values with an explicit "Unknown" category.
2. **Avoiding data leakage:** dropped the `work_interfere` column, which nearly reveals the target because people without a condition mostly left it blank.
3. **Modeling:** an 80/20 stratified train/test split; a decision tree tuned with 5-fold cross-validated grid search over depth and leaf size; and a random forest and a logistic regression baseline for comparison.
4. **Evaluation:** confusion matrices, accuracy, precision, recall, and F1 on the held-out test set.
5. **Interpretation:** random forest feature importance.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter / Google Colab

## How to run

Open the notebook in Google Colab and click **Runtime → Run all**. The first cell downloads the dataset automatically:

```python
import kagglehub
path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")
```

To run it locally instead: `pip install -r requirements.txt kagglehub`, then run the notebook top to bottom.

## Limitations

The survey is self-reported, from 2014, and mostly from US and UK respondents, so findings may not generalize. This model is meant to inform workplace policy, not to make decisions about individuals.

## Author

**Prosper C. Owunna**, M.S. Data Science candidate, Southern Arkansas University
