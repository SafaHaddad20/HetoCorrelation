

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score




data = pd.read_excel(r'C:\Users\umet\Desktop\herg\508_hetatoms_CANVAS.xlsx')
data_bits = pd.read_excel(r'C:\Users\umet\Desktop\herg\Fingerprints_bits.xlsx')

data_bits = data_bits.iloc[:508,:]
data_bits.drop('Structure', axis = 1, inplace = True)
data_merge = data_bits.merge(data, on=['pIC50', 'Name'])
Hetatoms_list= ['B','Br','Cl','F','Fe','N','P','S','Si','O']

data.drop(['Structure','Molecular_formula','IC50','C','H','Num heavy atoms'], axis=1, inplace=True)
sns.set_style('darkgrid')
plt.figure(figsize = (15,8))
for i in Hetatoms_list:
    sns.lmplot(x='Hetatoms', y= 'pIC50', hue = i, data = data, palette = 'rainbow')

plt.figure(figsize = (25,10))
plt.xlim(0,20)
sns.boxplot(x='Hetatoms', y= 'pIC50', hue ='O' ,saturation=0.75, width=0.8, data = data, palette= 'Spectral')

# data.drop(['B','Br','Fe','P','Si'],axis=1, inplace=True)
# data.drop(['Hetatoms'],axis=1, inplace=True)
# sns.heatmap(data.corr(), annot=True, cmap = 'Spectral')
# data.drop(['Cl','F','S'],axis=1, inplace=True)
# sns.heatmap(data.corr(), annot=True, cmap = 'Spectral')
# data.drop(['Name'],axis=1, inplace=True)

sns.lmplot(x='O', y= 'pIC50', hue = 'Hetatoms', data = data, palette = 'rainbow')

# sns.displot(data['O'],color='darkred')
# sns.displot(data['N'],color='darkred')
# sns.displot(data['pIC50'],color='darkred')

plt.figure(figsize = (25,10))
plt.xlim(0,20)
sns.boxplot(x='O', y= 'Fp_Linear', hue ='Hetatoms' ,saturation=0.75, width=0.8, data = data_merge, palette= 'Spectral')





# #________________________#
# x = data.drop(['pIC50'],axis=1)
# y = data['pIC50']

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

# model = linear_model.LinearRegression()

# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)

# my_dict= {'Actual': y_test, 'Pred':y_pred}
# compare = pd.DataFrame(my_dict)
# compare.sample(10)

# def error_scores(Actual,Pred):
#     MAE = mean_absolute_error(Actual,Pred)
#     MSE = mean_squared_error(Actual,Pred)
#     RMSE = np.sqrt(mean_squared_error(Actual,Pred))
#     score = r2_score(Actual,Pred)
#     return print('R2 = %f' %score)
# error_scores(y_test, y_pred)