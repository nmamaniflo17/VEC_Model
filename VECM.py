#!/usr/bin/env python
# coding: utf-8
---
title: VEC model to analyze the relationship between GDP and energy consumption in Peru, 1971 - 2014
subtitle: Application in Python
date: today
author:
  name: Nelson Brayan Mamani Flores
theme: journal
format:
  html:
    toc: true
    code-tools: true
jupyter: python3
---
# ### Abstract

# This paper presents empirical evidence on the relationship between energy consumption and GDP in Perú during the period 1971 to 2014. The results show that the series are not stationary, i.e., are individually I(1); in addition, we find a long-term relationship between both variables. Through a VECM we estimate short- and long-term elasticities to analyze the dynamics of adjustment. The results show that in the short-term the conservation hypothesis holds (i.e., no evidence of a short-term relationship of energy consumption to GDP). In the long-term, however, we find evidence of a feedback mechanism between both variables. Yet, this paper provides evidence that policymakers can implement policies aimed at energy conservation without hurting economic growth. 
# 
# **Key words:** Energy Consumption, Cointegration Relation, Vector Error Correction Model (VECM), Structural Break, Perú.

# ## 1. Introduction

# The oil crisis of the 1970s motivated researchers to study numerous relationships between macroeconomic variables. As a result, researchers began to study the relationships between macroeconomic variables such as oil prices, inflation, economic growth, energy consumption, the exchange rate, and other factors that affect the economy in general. This led to a greater understanding of how these variables are interconnected and how they can affect the economy of a country and the world as a whole. The oil crisis led to a significant increase in energy prices, which in turn had a major impact on the global economy. Also, since energy is a key input for production and consumption, it has become a central research topic in economics.
# 
# Given the importance between GDP dynamics and electricity consumption, this article will address the relationship between these variables through an error correction model. For this reason, the specific objective of this work is to provide empirical evidence of the existence or not of a strong relationship (in the short and long term) between energy consumption and GDP in Peru, as well as to analyze that, For the above reason, it is possible to implement environmental policies that promote the efficient use and conservation of energy. 
# 
# In this sense, this work is an empirical and applied contribution to the scarce literature on energy and its impact on production in Peru.

# ## 2.  The relationship between energy consumption and GDP

# The relationship between energy consumption and GDP is complex and multifaceted. Generally, there is a positive correlation between energy consumption and GDP, as countries with higher levels of economic development tend to consume more energy in order to power their industries, transportation systems, and households. Evidence also shows that it is possible to find a two-way causality between energy consumption and real GDP. In the study carried out by Al-Iriani (2006) for the six countries that make up the Gulf Cooperation Council (Kuwait, Oman, Saudi Arabia, Bahrain, the United Arab Emirates and Qatar), the results obtained indicate that there is unidirectional causality of GDP. . to energy consumption; Soytas, Sari and Ozdemir (2001) found evidence of a one-way causal relationship between energy consumption and GDP in Turkey from the cointegration method and error correction vector analysis.
# 
# A priori we could say that the relationship between energy consumption and GDP is complex and depends on a wide range of factors including the level of economic development of a country, its energy policies and the efficiency of its energy systems.

# ## 3. Econometric methodology and model

# It is known that to avoid obtaining misleading relationships in econometric estimates it is necessary to apply unit root tests to the series in order to determine the stationarity or not of the series. Then, if the series are non-stationary, that is, they have a unit root, it must be proved that they are cointegrated and, thus, have a long-term relationship between them.
# 
# Firstly, in this work the ADF (Augmented Dickey-Fuller) test and the KPSS (Kwiatkowski, Phillips, Smichdt and Shin) test were applied to test the order of integration of the series. Additionally, the Zivot-Andrews test was used to determine if the series are stationary or non-stationary in the presence of a possible unit root with structural break. 
# 
# As mentioned, if the series have a unit root (they are integrated of order one), according to Granger and Newbold (1974), the step to follow is to test cointegration between the series. In addition, the test proposed by Johansen based on the Lagrange Multiplier estimator (MLE) and the cointegration methodology proposed by Johansen and Juselius (1992) were used to prove the existence of a long-term relationship between the variables. The next step is to estimate the VAR Model (p) to determine the number of lags thereof based on the information criteria and thus determine the optimal lags of the VEC Model (p-1).

# ### 3.1.  Data and Application

# This is a time series study, which consists of annual time series of GDP and energy consumption, both in per capita terms. The series are obtained from the World Bank, the definition of energy consumption is: Electricity consumption measures the production of electricity generating plants and combined heat and electricity generation plants, minus transmission, distribution and transformation losses, and the own consumption of the plants. On the other hand, GDP per capita is measured at current prices.

# In[1]:


# import libraries

import pandas as pd 
import requests


# In[40]:


# Access the world bank database

from pandas_datareader import wb
KWH_PC= 'EG.USE.ELEC.KH.PC'     # Series Electrical energy consumption per capita (kWh)
PBI_PC= 'NY.GDP.PCAP.KN'        # GDP per capita series in USD (Constant prices)


# In[41]:


# Download the database with the selected variables

df1 = wb.download(indicator=[KWH_PC, PBI_PC], country='PER', start=1971, end=2014).sort_index()
df1.head()


# In[42]:


# Rename variables and remove country indexed column

data = df1.reset_index(level=['country'])
data = data.drop('country', axis=1).rename(columns={'EG.USE.ELEC.KH.PC': 'CE',
                                                    'NY.GDP.PCAP.KN': 'PBI'})
data.head()


# In[43]:


# Generate the logarithm of the variables

import numpy as np
data['LPBI'] = np.log(data['PBI'])
data['LCE'] = np.log(data['CE'])
data.head()


# In[6]:


# Generate the graph in time series of the logarithms of the variables

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth']=1.5

fig, ax = plt.subplots(figsize=(10, 6))
data[['LPBI','LCE']].plot(ax=ax)
ax.set_title('Figure1. Energy Consumption Per Capita and Real GDP Per Capita in Peru, 1971-2014')
ax.set_xlabel('Mes')
ax.set_ylabel('Valor')
plt.show()


# Graph 1 shows the series in logarithms for the study period and Graph 2 shows the first difference of the series in Graph 1. We can see that in graphs 1 and 2, the series are apparently correlated in some periods, but in others , power consumption increases.

# In[44]:


# Generate the first differences of the variables

data['DLPBI'] = (data['LPBI']- data['LPBI'].shift(1))*100
data['DLCE'] = (data['LCE']- data['LCE'].shift(1))*100
data.head()


# In[8]:


# Draw the first differences of the series

fig, ax = plt.subplots(figsize=(10, 6))
data[['DLPBI','DLCE']].plot(ax=ax)
ax.set_title('Figure2. First difference of energy consumption per capita and real GDP per capita in Perú, 1971-2014')
ax.set_xlabel('Año')
ax.set_ylabel('Valor')
plt.show()


# For example, in at least 10 periods, the series appear to go in opposite directions. This result of the preliminary analysis provides a first approximation that gives strength to the hypothesis of this study, which suggests that there is neither a short nor a long-term relationship between energy consumption and GDP, in both directions, that is, from energy consumption to GDP and vice versa. However, in recent years it can be seen that more movements in GDP translate into movements in energy consumption.

# In fact, if Graph 3 is observed, which presents a scatter graph with a linear trend, a first approximation can be seen, which indicates that the relationship between energy consumption and GDP is positive for the period of study in Peru.

# In[9]:


# Plot the dispersion between variables

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(data['LCE'], data['LPBI'])
fit = np.polyfit(data['LCE'], data['LPBI'], 1)
fit_fn = np.poly1d(fit)
plt.plot(data['LCE'], fit_fn(data['LCE']),'--r')
ax.set_title('Figure3. Dispersion between energy consumption per capita and real GDP per capita in Perú, 1971-2014')
ax.set_xlabel('LCE')
ax.set_ylabel('LPBI')
plt.show()


# ### 3.2. Unit Root Tests

# The Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test are both statistical tests commonly used in time series analysis to test for stationarity. Additionally, the Zivot-Andrews test was used to test for a structural break or regime shift in the data. A structural break occurs when there is a significant change in the underlying data generation process of a time series. The null hypothesis of the Zivot-Andrews test is that the time series is stationary and does not have a structural break. The alternative hypothesis is that the time series has a structural break at an unknown point in time. The results are shown below, taking into account deterministic components.

# In[10]:


# import libraries

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# In[11]:


# Dickey & Fuller test with constant for the LCE series

X = data["LCE"].values
result_adf1 = adfuller(X, regression='c')
print('ADF statistic: %f' % result_adf1[0])
print('p-value: %f' % result_adf1[1])
print('Critical values:')
for key, value in result_adf1[4].items():
    print('\t%s: %.3f' % (key, value))

if result_adf1[0] < result_adf1[4]["5%"]:
    print ("Reject the Ho - The time series is stationary")
else:
    print ("Do not reject Ho - The time series is not stationary")


# ##### Results analysis:

# The results indicate that we cannot reject the null hypothesis (Ho) that the time series is not stationary. This is because the ADF (Augmented Dickey-Fuller) value is greater than the critical values at the 1%, 5%, and 10% significance levels, and the p-value is greater than the 5% significance level. . Therefore, we do not have enough evidence to affirm that the time series is stationary. It is important to highlight that the non-stationarity of the time series can have important implications in its modeling and in making decisions based on it.

# In[12]:


# Dickey & Fuller test with constant and trend for the LCE series

X = data["LCE"].values
result_adf2 = adfuller(X, regression='ct')
print('ADF statistic: %f' % result_adf2[0])
print('p-value: %f' % result_adf2[1])
print('Critical values:')
for key, value in result_adf2[4].items():
    print('\t%s: %.3f' % (key, value))

if result_adf2[0] < result_adf2[4]["5%"]:
    print ("Reject the Ho - The time series is stationary")
else:
    print ("Do not reject Ho - The time series is not stationary")


# ##### Results analysis:

# The results indicate that we cannot reject the null hypothesis (Ho) that the LCE time series is not stationary. This is because the ADF (Augmented Dickey-Fuller) value is greater than the critical values at the 1%, 5%, and 10% significance levels, and the p-value is greater than the 5% significance level. . Therefore, we do not have enough evidence to affirm that the LCE time series is stationary. It is important to highlight that the non-stationarity of the time series can have important implications in its modeling and in making decisions based on it. In addition, when using the test with constant and trend, the possible presence of a trend in the time series is taken into account, which may affect its stationarity.

# In[13]:


# Dickey & Fuller test with constant for the lPBI series

Y = data["LPBI"].values
result_adf3 = adfuller(Y, regression='c')
print('ADF statistic: %f' % result_adf3[0])
print('p-value: %f' % result_adf3[1])
print('Critical values:')
for key, value in result_adf3[4].items():
    print('\t%s: %.3f' % (key, value))

if result_adf3[0] < result_adf3[4]["5%"]:
    print ("Reject the Ho - The time series is stationary")
else:
    print ("Do not reject Ho - The time series is not stationary")


# ##### Results analysis:

# The results indicate that we cannot reject the null hypothesis (Ho) that the lGDP time series is not stationary. This is because the ADF (Augmented Dickey-Fuller) value is greater than the critical values at the 1%, 5%, and 10% significance levels, and the p-value is greater than the 5% significance level. . Therefore, we do not have sufficient evidence to affirm that the lGDP time series is stationary. It is important to highlight that the non-stationarity of the time series can have important implications in its modeling and in making decisions based on it. Using the constant test assumes that the time series has no deterministic trend, but may have a constant.

# In[14]:


# Dickey & Fuller test with constant and trend for the lPBI series

Y = data["LPBI"].values
result_adf4 = adfuller(Y, regression='ct')
print('ADF statistic: %f' % result_adf4[0])
print('p-value: %f' % result_adf4[1])
print('Critical values:')
for key, value in result_adf4[4].items():
    print('\t%s: %.3f' % (key, value))

if result_adf4[0] < result_adf4[4]["5%"]:
    print ("Reject the Ho - The time series is stationary")
else:
    print ("Do not reject Ho - The time series is not stationary")


# ##### Results analysis:

# The results indicate that we cannot reject the null hypothesis (Ho) that the IPBI time series is not stationary. This is because the ADF (Augmented Dickey-Fuller) value is greater than the critical values at the 1%, 5%, and 10% significance levels, and the p-value is greater than the 5% significance level. . Therefore, we do not have enough evidence to affirm that the GDPl time series is stationary. When using the test with constant and trend, the possible presence of a trend and a constant in the time series is taken into account, which may affect its stationarity. It is important to highlight that the non-stationarity of the time series can have important implications in its modeling and in decision-making based on it.

# In[15]:


# import libraries

from statsmodels.tsa.stattools import kpss


# In[16]:


import warnings
warnings.filterwarnings("ignore")


# In[17]:


# Kwiatkowski–Phillips–Schmidt–Shin test with constant for the lCE series

result_kpss1 = kpss(X, regression='c')
print('KPSS Statistic: %f' % result_kpss1[0])
print('p-value: %f' % result_kpss1[1])
print('Critical values:')
for key, value in result_kpss1[3].items():
    print('\t%s: %.4f' % (key, value))

if result_kpss1[0] < result_kpss1[3]["5%"]:
    print ("Reject the Ho - The time series is not stationary")
else:
    print ("Do not reject Ho - The time series is stationary")


# ##### Results analysis:

# LCE series indicate that we cannot reject the null hypothesis (Ho) that the time series is not stationary. This is because the value of the KPSS statistic is greater than the critical values at all significance levels, and the p-value is less than the 1% significance level.

# In[18]:


# Kwiatkowski–Phillips–Schmidt–Shin test with constant and trend for the lCE series

result_kpss2 = kpss(X, regression='ct')
print('KPSS Statistic: %f' % result_kpss2[0])
print('p-value: %f' % result_kpss2[1])
print('Critical values:')
for key, value in result_kpss2[3].items():
    print('\t%s: %.4f' % (key, value))

if result_kpss2[0] < result_kpss2[3]["5%"]:
    print ("Reject the Ho - The time series is not stationary")
else:
    print ("Do not reject the Ho - The time series is stationary")


# ##### Results analysis:

# The results of the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test with constant and trend for the ICE series indicate that we cannot reject the null hypothesis (Ho) that the time series is not stationary. This is because the value of the KPSS statistic is less than the critical values at all significance levels, and the p-value is less than the 1% significance level.
# 
# When using the test with constant and trend, the possible presence of a trend and a constant in the time series is taken into account, which may affect its stationarity.
# 
# Therefore, the results suggest that the lCE series has a deterministic trend, which means that its mean is not constant over time. This can have important implications for time series modeling and decision making, as a trend can affect the prediction and interpretation of the results. It is important to take these results into account when performing analysis and modeling of the lCE time series.

# In[19]:


# Kwiatkowski–Phillips–Schmidt–Shin test with constant for the lCE series

result_kpss3 = kpss(Y, regression='c')
print('KPSS Statistic: %f' % result_kpss3[0])
print('p-value: %f' % result_kpss3[1])
print('Critical values:')
for key, value in result_kpss3[3].items():
    print('\t%s: %.4f' % (key, value))

if result_kpss3[0] < result_kpss3[3]["5%"]:
    print ("Reject the Ho - The time series is not stationary")
else:
    print ("Do not reject the Ho - The time series is stationary")


# ##### Results analysis:

# The results of the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test with constant for the LCE series indicate that we can reject the null hypothesis (Ho) that the time series is not stationary. This is because the value of the KPSS statistic is greater than the critical values at all significance levels, and the p-value is greater than the 5% significance level.
# 
# This suggests that the LCE series is stationary in the sense that its mean and variance are constant over time, and that it does not have a deterministic trend. This is important in time series modeling and analysis, as a stationary series may be easier to model and predict.

# In[20]:


# Kwiatkowski–Phillips–Schmidt–Shin test with constant and trend for the lCE series

result_kpss4 = kpss(Y, regression='ct')
print('KPSS Statistic: %f' % result_kpss4[0])
print('p-value: %f' % result_kpss4[1])
print('Critical values:')
for key, value in result_kpss4[3].items():
    print('\t%s: %.4f' % (key, value))

if result_kpss4[0] < result_kpss4[3]["5%"]:
    print ("Reject the Ho - The time series is not stationary")
else:
    print ("Do not reject the Ho - The time series is stationary")


# ##### Results analysis:

# The results of the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test with constant and trend for the ICE series indicate that we cannot reject the null hypothesis (Ho) that the time series is not stationary. This is because the value of the KPSS statistic is less than the critical values at all significance levels, and the p-value is less than the 5% significance level.
# This suggests that the ICE series is not stationary in the sense that it has a deterministic trend. It is possible that the series is stationary in the sense that its mean and variance are constant over time, but this cannot be determined with the KPSS test with constant and trend.

# The results of the unit root tests for both series, there it can be seen that the estimated statistics are less than the critical values at 5% significance. Therefore, it is concluded that the series are integrated of order one in levels and of order zero (stationary) in differences. Additionally, deterministic components such as intercept and trend are included in the tests.

# In[21]:


# Graph both series separately

fig, ax = plt.subplots(figsize=(15,5), nrows=1, ncols=2)
ax[0].plot(data['LPBI'])
ax[1].plot(data['LCE'])
ax[0].set_title("LBI series chart")
ax[1].set_title("LCE series chart")
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=90)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
plt.show()


# In this case, to carry out the Zivot Andrews test, the trend of our series was taken into account; since it captures individually and precisely the structural break of the series. However, using other tests, multiple structural breaks could be obtained, which would make the cointegration task more complex. That is why we only take the trend in our series to identify the breaks produced in the historical evolution of the series.

# In[22]:


# import libraries

from statsmodels.tsa.stattools import zivot_andrews


# In[23]:


# Zivot andrews test with trend for the lPBI series

result_za1 = zivot_andrews(data['LPBI'], trim = 0.15, maxlag=5, regression='t')
print('Statistician Zivot Andrews: %f' % result_za1[0])
print('P-value: %f' % result_za1[1])
print('Critical values:') 
for key, value in result_za1[2].items():
    print('\t%s: %.3f' % (key, value))
print('Baselag: %f' % result_za1[3])
year = data.iloc[result_za1[4]].name
print(f"Year corresponding to the Breakpoint index: {year}")

if result_za1[0] < result_za1[2]["5%"]:
    print ("Reject the Ho - The series is stationary in trend")
else:
    print ("Do not reject Ho - The series has a unit root with only one structural break")


# ##### Results analysis

# The null hypothesis of this test is that there is a unit root in the time series with a structural break at some point. In this case, the test returned a statistical value of -4.282785 and a p-value of 0.070014.
# 
# Critical values are used to compare with the statistical value and determine whether or not to reject the null hypothesis. In this case, the statistical value is less than the corresponding critical value at the 5% level of significance but not at 1%, indicating that there is insufficient evidence to reject the null hypothesis at the 5% level of significance. However, if a 1% significance level is considered, the statistical value is greater than the corresponding critical value, indicating that there is sufficient evidence to reject the null hypothesis at the 1% significance level.
# 
# Therefore, it is concluded that the time series has a unit root with a single structural break in the year 1997 (index 26). It is important to note that the series does not have a stationary trend and that future values of the series may depend on past values. This should be considered when performing analyzes and projections based on the time series in question.

# In[24]:


# Zivot andrews test with trend for the LCE series

result_za2 = zivot_andrews(data['LCE'], trim = 0.15 , maxlag=5, regression='t')
print('Zivot-Andrews statistic: %f' % result_za2[0])
print('P-value: %f' % result_za2[1])
print('Critical values:') 
for key, value in result_za2[2].items():
    print('\t%s: %.3f' % (key, value))
print('Baselag: %f' % result_za2[3])
year = data.iloc[result_za2[4]].name
print(f"Year corresponding to the Breakpoint index: {year}")

if result_za2[0] < result_za2[2]["5%"]:
    print ("Reject the Ho - The series is stationary in trend")
else:
    print ("Do not reject Ho - The series has a unit root with only one structural break")


# ##### Results analysis:

# The null hypothesis of this test is that there is a unit root in the time series with a structural break at some point. In this case, the test returned a statistical value of -3.112052 and a p-value of 0.587514.
# 
# Critical values are used to compare with the statistical value and determine whether or not to reject the null hypothesis. In this case, the statistical value is not less than the critical values corresponding to the significance level of 1%, 5%, and 10%, indicating that there is insufficient evidence to reject the null hypothesis. Therefore, the time series is considered to have a unit root with a single structural break in the year 1996 (index 25).
# 
# In summary, it cannot be affirmed that the series has a stationary trend, which implies that the future values of the series can depend on the past values. It is important to keep this in mind when performing analyzes and projections based on the time series in question.

# In[25]:


# Plot both series with the structural break

fig, ax = plt.subplots(figsize=(15,5), nrows=1, ncols=2)
ax[0].plot(data['LPBI'])
ax[1].plot(data['LCE'])
ax[0].set_title("LBI series chart with structural break")
ax[1].set_title("LCE series chart with structural break ")
ax[0].axvline(x='1997', color='g', linestyle='--')
ax[1].axvline(x='1996', color='r', linestyle='--')
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=90)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
plt.show()


# These results indicate that the null hypothesis of unit root for the series in levels cannot be rejected at any conventional level. The test calculates the breaks that occurred in both series and both are consistent with historical events, such as the privatization process of the electricity sector, which began in 1994 with the sale of distribution companies in Lima, continuing with the sale of generating companies in 1995 and 1996. Likewise, with respect to GDP, the results coincide with the effects of the El Niño phenomenon of 1997-98, which reached great intensity in Peru. In this regard, Contreras et. al (2016), indicate that the El Niño phenomenon constitutes a risk due to a supply shock for the Peruvian economy. When El Niño reaches extraordinary magnitudes, it destroys part of the economy's capital stock and affects the flow of production of goods and services, all of which generates impacts on potential GDP, amplifying business cycles.

# In this sense, to capture the presence of these two structural breaks in our model, we will generate dummy variables for each year in question.

# In[45]:


# Create dummy variables to identify structural breaks

data['fecha'] = pd.date_range(start='1971-01-01', end='2014-01-01', freq='AS')
data['dummy_LPBI'] = np.where(data['fecha'].dt.year >= 1997, 0, 1)
data['dummy_LCE'] = np.where(data['fecha'].dt.year >= 1996, 0, 1)
data.head(30)


# In[27]:


# Import libraries

from statsmodels.tsa.vector_ar.var_model import VAR


# In[28]:


data1 = data.iloc[:, [2,3]]


# In[46]:


# Get the order of lags

model = VAR(data1)
for i in [1,2,3,4,5]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# In[47]:


# Select the order of lags

x = model.select_order(maxlags=5)
x.summary()


# These results refer to the selection of the order of lags for a VAR (Vector Autoregression) model using four information criteria: AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), FPE (Final Prediction Error) and HQIC (Hannan- Quinn Information Criterion).
# 
# Several lag orders have been tested, from 0 to 5. The lowest values of the information criteria (AIC, BIC, FPE, HQIC) indicate that that lag order is the best for the VAR model.
# 
# In this case, the selected lag order is 2, since it has the lowest BIC value of all the lag orders tested. In addition, its AIC, FPE, and HQIC value are very close to the minimum value, so it can also be considered as a good option.

# ### 3.3. Cointegration Test

# As mentioned above, the Johansen test was applied. Below are the results of the Maximum Likelihood Cointegration tests based on the range of the **&Pi;** matrix, the Trace and Maximum Eigenvalue tests. The results of the Cointegration test between real GDP and energy consumption imply that the null hypothesis of no Cointegration in both tests, that is, H<sub>0</sub> is rejected at 5% significance, since the is statistic calculated for each is greater than the critical value.

# In[48]:


# Import libraries

from statsmodels.tsa.vector_ar.vecm import coint_johansen


# In[49]:


# Method 1: Perform Johansen's cointegration test

data1 = data.iloc[:, [2, 3]]
array = np.array(data1.values)
result_cj = coint_johansen(array, det_order=-1, k_ar_diff=1)
print('Rank:', result_cj.ind[0])
print('Eigenvalue:', result_cj.eig[0])
print('trace statistic: ', result_cj.trace_stat[0])
print('Critical value at 5%: ', result_cj.trace_stat_crit_vals[0][1])
print('Max Eigenvalue: ', result_cj.max_eig_stat[0])
print('Max Critical value at 5%: ', result_cj.max_eig_stat_crit_vals[0][1])
#
#
print('Rank:', result_cj.ind[1])
print('Eigenvalue:', result_cj.eig[1])
print('trace statistic: ', result_cj.trace_stat[1])
print('Critical value at 5%: ', result_cj.trace_stat_crit_vals[1][1])
print('Max Eigenvalue: ', result_cj.max_eig_stat[1])
print('Max Critical value at 5%: ', result_cj.max_eig_stat_crit_vals[1][1])


# Based on the result obtained with the Johansen Cointegration Test, the Error Correction Model is derived. In this part, we include in the Error Correction Model, two dummy intervention variables that collect the effects of structural breaks in the variables found with the test of Zivot-Andrews, to take them into account in the short- and long-term adjustment dynamics. 
# 
# The Granger representation theorem allows us to model short-term and long-term dynamics through an error correction model. However, first the optimal number of lags must be determined, which is obtained through the VAR(p) model. In this case, the optimal number of lags is 2, that is, our VAR model is of order 2, VAR (2). Now, the error correction model (VEC(p-1)) will be a VEC of order 1, that is, VEC(1).

# In[50]:


# Import libraries

from statsmodels.tsa.vector_ar import vecm


# In[51]:


# Method 2: Perform Johansen's cointegration test

vec_rank=vecm.select_coint_rank(data1, det_order=-1, k_ar_diff=1, method='trace', signif=0.05)
print(vec_rank.summary())


# The result of the Johansen cointegration test indicates that for r = 0, the value of the trace test statistic is 19.63. The corresponding critical value for a significance level of 5% is 12.32. Since the value of the test statistic is greater than the critical value, one can reject the null hypothesis that there is no cointegration between the variables and conclude that there is at least one cointegration relationship between them.

# In[52]:


# Get maximum eigenvalues

vec_rank1=vecm.select_coint_rank(data1, det_order=-1, k_ar_diff=2, method='maxeig', signif=0.05)
print(vec_rank1.summary())


# In[53]:


dummies = data.iloc[:, [7, 8]]
array = np.array(dummies.values)


# Based on the VEC model, after applying the Johansen Cointegration Test between energy consumption and GDP, the adjustment dynamics in the short and long term can be analyzed.

# In[54]:


# Import Libraries

from statsmodels.tsa.vector_ar.vecm import VECM


# In[55]:


# Apply the error correction model (VECM)

vecm = VECM(data1, exog=dummies, k_ar_diff=1, coint_rank=1, deterministic='n', dates=data['fecha'], )
vecm_fit = vecm.fit()
print(vecm_fit.summary())


# The results show that in the short term, the L1.LPBI coefficient is positive and significant at the 1% level, which indicates that in the short term, a variation in the GDP growth rate (LPBI) will translate into a increase in its own growth rate in the same period. On the other hand, the L1.LCE coefficient is negative but not statistically significant, which suggests that the growth rate of electricity consumption (LCE) does not significantly affect the GDP growth rate in the short term.
# 
# In this order of ideas, the energy conservation hypothesis is fulfilled for the Peruvian economy in the short term. Regarding the long-term relationships, the results of the cointegration relationships can be analyzed in the last table. Where we can see that the beta.2 coefficient is negative and significant at the 1% level, which indicates that in the long term there is a direct relationship between the growth rate of Peru's GDP and the growth rate of energy consumption in Peru . In other words, in the long term, an increase in energy consumption of 1% generates an increase in GDP of 1,256%, therefore, in the long term, energy consumption does affect GDP.
# 
# Finally, the adjustment coefficient (speed of adjustment) to long-term imbalances of the model (∆LPIB) is 2.75%, while which for the model (∆LCE) is 8.12%. This implies a faster adjustment in the energy consumption error correction model.

# In[56]:


# Make forecasts

vecm1 = VECM(data1, k_ar_diff=2, coint_rank=1, deterministic='n', dates=data['fecha'])
vecm_fit1 = vecm1.fit()
vecm_fit1.plot_forecast(5)


# ## 4. Conclusiones

# In this article, the short- and long-term relationship between energy consumption and real GDP for the Colombian economy during the period 1970-2009 was analyzed, using annual time series. First, the long-term elasticity of energy consumption to real GDP was estimated and then, under the VEC modeling, the short- and long-term dynamics were recognized.
# 
# The empirical results for the Peruvian case suggest the existence of a long-term bidirectional causal relationship between energy consumption and GDP. In other words, the fact that there is cointegration between the variables confirms the relationship between them, that is, it suggests that in the long term there is a feedback between energy consumption and GDP.
# 
# In summary, the results indicate that in the short term there is no relationship between energy consumption and GDP, but there is in the long term, so our results can be placed under the long-term energy conservation hypothesis. term and a long-term feedback effect.

# ## References
# - Abosedra, A., et al. (1991). New evidence on the causal relationship between U.S. Energy consumption y Gross National product. Journal of Energy and Development.
# - Al-iriani, M. (2006). Energy GDP relationship revisted: an example from GCC countries using panel causality. Energy Policy.
# - Dickey, D. & Fuller, W. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root. Journal of the American Statistical Association.
# - Erol, et al. (2001). On the causal relationship between energy and income for industrialized countries. Journal of Energy Development.
# - Granger, C. & Newbold, P. (1974). Spurious Regressions in Econometrics. Journal of Econometrics, vol. 2: 111-120, 1974.
# - Hwang, D. & Gum, B. (1992). The causal relationship between energy and GNP: the case of Taiwan. Journal of Energy and Development.
# - Johansen, S. (1988). Statistical analysis of cointegration vectors. Journal of Economic Dynamics and Control, vol. 12.
# - Ozturk et at. (2010). The causal relationship between energy consumption and GDP in Albania, Bulgaria, Hungary and Romania: Evidence from ARDL bound testing approach. Applied Energy, Elsevier, vol. 87.
# - Sims, C. (1980). Macroeconomics and Reality. Econométrica, vol. 48(1): 1-48.
# - Soytas et al. (2001). Energy Consumption and GDP Relations in Turkey: A Cointegration and Vector Error Correction Analysis. Economics and Business in Transition: Facilitating Competitiveness and Change in the Global Environment Proceedings. Global Business and Technology Association. 
# - Yu et al. (1984). The relationship between energy and GNP: further results. Energy Economics, vol. 6: 186-190, 1984.
# 
