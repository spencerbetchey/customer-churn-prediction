# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to cancel their 
service, built as a way for me to learn the data science workflow hands on, taken from messy raw data 
all the way to a live interactive webpage.

**[Try the live app here](https://customer-churn-prediction-spencer.streamlit.app/)**

## Why I built this

Over this past summer, I took an interest in data science, and wanted to explore what a data science project
actually looks like. As someone with very minimal data science knowledge/experience, I planned this
project out over 3-4 weeks and worked through it step by step with help from Claude AI, using it as a mentor
to explain concepts before I used them, ask me to reason through decisions, and pushing back when my
assumptions turned out to be wrong. This README reflects what I actually learned, including a few
results that surprised me. Overall, it was a great learning experience. 

## What it does

Given some basic info about a customer: how long they've been with the company, what services 
they have, their contract type, monthly bill, etc, the app predicts how likely they are to churn 
(cancel their service), and shows that as a Low/Medium/High risk level along with the actual calculated churn %.

## The dataset

I used the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 
a public dataset of about 7,000 telecom customers, with a mix of demographic info, account details, 
and which services each customer subscribes to.

## What I did

1. **Cleaned the data** - Found and fixed a bug where 11 customers had a blank value 
   instead of `$0` for `TotalCharges` (all brand new customers with 0 months of tenure). Also 
   dropped a customer ID column that had no real predictive value, and encoded all the categorical 
   columns (like Yes/No fields and multi option fields like Contract type) into numbers a model 
   can actually use (Mainly 0s and 1s to represent yes/no for fields).

2. **Explored the data** - before building anything, I looked for patterns by hand. Turns out 
   customers with shorter tenure and month to month contracts tend to churn a lot more, which matched 
   my initial guess. But some other assumptions I made going in (like guessing gender or having a 
   partner would matter a lot) turned out to be weaker predictors.

3. **Trained and compared 3 models** - Logistic Regression, Random Forest, and XGBoost. Honestly, 
   I expected the more complex models to win. They didn't. The simple Logistic Regression baseline 
   ended up outperforming both, even after tuning the XGBoost model with GridSearchCV. That was a genuinely 
   useful lesson: more complexity isn't automatically better, especially on a dataset where the 
   relationships between features and churn are fairly linear.

4. **Explained the model's predictions with SHAP** - instead of just trusting a black box 
   probability score, I used SHAP to see exactly which factors pushed a specific customer's risk 
   up or down. One surprising find was that higher monthly charges actually pushed predictions *away* 
   from churn, which went against my original assumption that pricier customers leave more often. This is 
   likely because it's tangled up with other features like having fiber internet.

5. **Built and deployed a Streamlit app** - so anyone can plug in hypothetical customer info and 
   get a real prediction back.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 80.7% | 65.9% | 56.4% | **60.8%** |
| Random Forest | 78.9% | 62.8% | 49.7% | 55.5% |
| XGBoost (tuned) | 80.3% | 65.9% | 53.2% | 58.9% |

Logistic Regression came out on top on F1 score, which is the metric I cared about most here since 
plain accuracy is misleading on this dataset. About 73% of customers don't churn, so a model that 
never predicts churn would already score 73% accuracy while being completely useless.

## Limitations / Things I'd improve next

- `TotalCharges` is basically `tenure × MonthlyCharges`, so having all three in the model probably 
  waters down feature importance readings (I confirmed a 0.83 correlation between tenure and 
  TotalCharges). Dropping the redundant column would likely give a cleaner importance ranking.
- I only did a light hyperparameter search on XGBoost. A more thorough search might close the gap 
  with Logistic Regression, or even beat it.
- The app currently uses the default 50% probability threshold to decide "churn" vs "no churn." In 
  a real business setting, that threshold would probably be tuned based on how costly a missed 
  churner is versus a wasted retention offer.

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Jupyter

## Running it locally

```bash
git clone https://github.com/spencerbetchey/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\Activate.ps1  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
streamlit run app/app.py
```