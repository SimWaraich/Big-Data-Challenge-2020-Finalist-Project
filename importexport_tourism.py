import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans

from internationaltourism import tourismcovid

tourismcovid = tourismcovid[['country', 'population', 'tourists_per_cap', 'log_tourists_per_cap', 'deaths_per_cap',
                             'log_deaths_per_cap']]
tourismcovid.columns = ['country', 'population', 'inbound_per_cap', 'log_inbound_per_cap', 'deaths_per_cap',
                        'log_deaths_per_cap']

employees = pd.read_csv('tourism_employees.csv')
employees = employees[employees['employees'] != '..'].reset_index()
employees = employees.drop(labels=['index'], axis=1)
employees['employees'] = employees['employees'].str.replace(',', '')
employees['employees'] = pd.to_numeric(employees['employees'])

tourismcovid = tourismcovid.merge(employees, how='inner')
tourismcovid['employees_per_cap'] = (tourismcovid['employees'] / tourismcovid['population']) * 100000
tourismcovid['log_employees_per_cap'] = np.log(tourismcovid['employees_per_cap'])

hotels = pd.read_csv('hotels.csv')
hotels = hotels[hotels['hotels'] != '..'].reset_index()
hotels = hotels.drop(labels=['index'], axis=1)
hotels['hotels'] = hotels['hotels'].str.replace(',', '')
hotels['hotels'] = pd.to_numeric(hotels['hotels'])

tourismcovid = tourismcovid.merge(hotels, how='inner')
tourismcovid['hotels_per_cap'] = (tourismcovid['hotels'] / tourismcovid['population']) * 100000
tourismcovid['log_hotels_per_cap'] = np.log(tourismcovid['hotels_per_cap'])

outbound = pd.read_csv('outbound.csv')
outbound = outbound[outbound['outbound'] != '..'].reset_index()
outbound = outbound.drop(labels=['index'], axis=1)
outbound['outbound'] = outbound['outbound'].str.replace(',', '')
outbound['outbound'] = pd.to_numeric(outbound['outbound'])

tourismcovid = tourismcovid.merge(outbound, how='inner')
tourismcovid['outbound_per_cap'] = (tourismcovid['outbound'] / tourismcovid['population']) * 100000
tourismcovid['log_outbound_per_cap'] = np.log(tourismcovid['outbound_per_cap'])

balance = pd.read_csv('balance.csv')
balance = balance[balance['balance'] != '..'].reset_index()
balance = balance.drop(labels=['index'], axis=1)
balance['balance'] = balance['balance'].str.replace(',', '')
balance['balance'] = pd.to_numeric(balance['balance'])

tourismcovid = tourismcovid.merge(balance, how='inner')
# print(tourismcovid)

X = tourismcovid[['inbound_per_cap', 'outbound_per_cap']].values

model = KMeans(n_clusters=4)
model.fit(X)
labels = model.predict(X)
# print(labels)

num_clusters = list(range(1, 9))
inertias = []

# for k in num_clusters:
#     model = KMeans(n_clusters=k)
#     model.fit(X)
#     inertias.append(model.inertia_)
#
# plt.plot(num_clusters, inertias, '-o')
#
# plt.xlabel('number of clusters (k)')
# plt.ylabel('inertia')
#
# plt.show()

tourismcovid['labels'] = labels

# sns.barplot(data=tourismcovid, x='labels',y='deaths_per_cap')
# plt.show()

label_0 = tourismcovid[tourismcovid['labels'] == 0].deaths_per_cap
label_1 = tourismcovid[tourismcovid['labels'] == 1].deaths_per_cap
label_2 = tourismcovid[tourismcovid['labels'] == 2].deaths_per_cap
label_3 = tourismcovid[tourismcovid['labels'] == 3].deaths_per_cap

fstat, pval = f_oneway(label_0, label_1, label_2, label_3)
# print(pval)

sns.scatterplot(data=tourismcovid, x='inbound_per_cap', y='outbound_per_cap', hue='labels', legend='full')
# plt.show()
