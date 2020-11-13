from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from domestictourism import covidfactors

X = covidfactors['log_outbound_per_cap'].values.reshape(-1,1)
y = covidfactors['log_deaths_per_cap'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

slope, intercept, r_value, p_value, std_err = stats.linregress(covidfactors['log_outbound_per_cap'],
                                                               covidfactors['log_deaths_per_cap'])
# r_value decent, p_value insignificant
# print(r_value, p_value)

# figure out what these values represent
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)

X = covidfactors[['log_outbound_per_cap', 'log_income', 'log_le', 'log_me']].values
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
