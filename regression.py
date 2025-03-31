import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf  # Import statsmodels.formula.api

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np # Import numpy

# Read in training, testing, and "new site" data sets
#   1. Click the file folder icon (bottom-most icon on the left)
#   2. Click the page icon with an up arrow (leftmost icon under the word "Files")
#   3. Choose the correct CSV files and click Open
train = pd.read_csv("SITE-DATA-TRAIN.csv")
test = pd.read_csv("SITE-DATA-TEST.csv")
new = pd.read_csv("SITE-DATA-48-STORES-UNDER-CONST.csv")

# Original model:
original_variables = ['AGINC', 'SQFT', 'COM60', 'COLGRAD']
# JJJ original_model = sm.OLS(train['ANNUAL PROFIT'], sm.add_constant(train[original_variables]))
original_model = smf.ols('Q("ANNUAL PROFIT") ~ AGINC + SQFT + COM60 + COLGRAD', data=train)

original_result = original_model.fit()
print(original_result.summary())

# Original model R^2 (training set and testing set): jjj
original_baseline = train['ANNUAL PROFIT'].mean()
original_train_prediction = original_result.predict(sm.add_constant(train[['AGINC', 'SQFT', 'COM60', 'COLGRAD']]))
original_test_prediction = original_result.predict(sm.add_constant(test[['AGINC', 'SQFT', 'COM60', 'COLGRAD']]))
training_R2 = 1 - ((original_train_prediction - train['ANNUAL PROFIT'])*2).sum() / ((original_baseline - train['ANNUAL PROFIT'])*2).sum()
testing_R2 = 1 - ((original_test_prediction - test['ANNUAL.PROFIT'])*2).sum() / ((original_baseline - test['ANNUAL.PROFIT'])*2).sum()

# Additional regression statistics
n = len(train)  # sample size
p = len(original_variables)  # number of predictors (excluding intercept)

# Multiple R (correlation coefficient) - square root of R-squared
multiple_R = np.sqrt(original_result.rsquared)

# Standard Error of the regression (also called the standard error of the estimate)
standard_error = np.sqrt(original_result.mse_resid)

# Adjusted R-squared - statsmodels already calculates this, but including explicitly
adjusted_R_squared = original_result.rsquared_adj

# Print additional regression statistics
print("\n--- Additional Regression Statistics ---")
print(f"Multiple R: {multiple_R:.6f}")
print(f"Standard Error: {standard_error:.6f}")
print(f"Adjusted R-squared: {adjusted_R_squared:.6f}")
print(f"Observations: {n}")



# Original profit prediction:
original_profit = original_result.predict(sm.add_constant(new[['AGINC', 'SQFT', 'COM60', 'COLGRAD']])).sum()
print("Original Profit (Sum of Predictions):", original_profit)

# Display the individual predictions before summing
# predictions = original_result.predict(sm.add_constant(new[['AGINC', 'SQFT', 'COM60', 'COLGRAD']]))
# print("\nIndividual predictions:")
# print(predictions)

# Display additional statistics
print("\nAverage prediction:", predictions.mean())
print("Number of predictions:", len(predictions))
print("Min prediction:", predictions.min())
print("Max prediction:", predictions.max())

# ----------------- ANOVA ANALYSIS -----------------

# 1. ANOVA table from regression model
# This shows the significance of the model as a whole
print("\n--- ANOVA from Regression Model ---")
anova_table = sm.stats.anova_lm(original_result)
print(anova_table)

# 2. Calculate Sum of Squares
y = train['ANNUAL PROFIT']
y_hat = original_train_prediction
y_bar = y.mean()

# Total Sum of Squares (TSS)
TSS = ((y - y_bar)**2).sum()
# Regression Sum of Squares (RSS) or Explained Sum of Squares (ESS)
ESS = ((y_hat - y_bar)**2).sum()
# Error Sum of Squares (SSE) or Residual Sum of Squares
SSE = ((y - y_hat)**2).sum()

print("\n--- Sum of Squares Breakdown ---")
print(f"Total Sum of Squares (TSS): {TSS:.2f}")
print(f"Explained Sum of Squares (ESS): {ESS:.2f}")
print(f"Residual Sum of Squares (SSE): {SSE:.2f}")
print(f"ESS + SSE = {ESS + SSE:.2f}, should approximately equal TSS = {TSS:.2f}")

# 3. F-test
# The F-statistic tests whether the model as a whole is significant
# This is automatically calculated in the ANOVA table, but we can do it manually too
n = len(y)  # sample size
p = len(original_variables)  # number of predictors
F = (ESS/p) / (SSE/(n-p-1))
p_value = 1 - stats.f.cdf(F, p, n-p-1)

print("\n--- Manual F-Test ---")
print(f"F-statistic: {F:.4f}")
print(f"p-value: {p_value:.10f}")
print(f"Degrees of freedom: {p}, {n-p-1}")

# 4. Individual predictor significance
# This is similar to the t-tests shown in the regression summary

# 5. Optional: One-way ANOVA if you have categorical variables
# For example, if you want to see if profits differ by region (assuming you have a 'REGION' column)
if 'REGION' in train.columns:
    print("\n--- One-way ANOVA by Region ---")
    regions = train['REGION'].unique()
    groups = [train[train['REGION'] == region]['ANNUAL PROFIT'] for region in regions]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.10f}")
