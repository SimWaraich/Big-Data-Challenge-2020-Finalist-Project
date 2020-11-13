import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from internationaltourism import tourismcovid

ipp = pd.read_csv('income_per_person.csv')
ipp = ipp[['country', '2020']]
ipp.columns = ['country', 'income']
ipp['log_income'] = np.log(ipp['income'])

ley = pd.read_csv('life_expectancy_years.csv')
ley = ley[['country', '2020']]
ley.columns = ['country', 'life_expectancy']
ley['log_le'] = np.log(ley['life_expectancy'])

may = pd.read_csv('median_age_years.csv')
may = may[['country', '2020']]
may.columns = ['country', 'median_age']
may['log_me'] = np.log(may['median_age'])

tc2 = tourismcovid[['country', 'log_deaths_per_cap', 'tourists_per_cap', 'log_tourists_per_cap']]

df = ipp.merge(ley, how='inner')
factors = df.merge(may, how='inner')

# print(set(tc2.columns).intersection(set(factors.columns)))
covidfactors = tc2.merge(factors, how='inner')

# plt.scatter(covidfactors['log_me'], covidfactors['log_deaths_per_cap'])
# plt.show()

X = covidfactors[['log_tourists_per_cap', 'log_income', 'log_le', 'log_me']].values
y = covidfactors['log_deaths_per_cap'].values

# plt.scatter(covidfactors['log_tourists_per_cap'], covidfactors['log_deaths_per_cap'])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# print(regressor.coef_)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)

F, pval = f_regression(X, y.ravel())
# print(F, pval)

mreg = sm.OLS(y, X).fit()
# print(mreg.summary())
