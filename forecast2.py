
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import warnings
import streamlit as st


warnings.filterwarnings("ignore")
co2data = pd.read_csv("co2dataset.csv")

# converting string data to datetime
co2data['Year'] = pd.to_datetime(co2data['Year'],format='%Y', errors='ignore')
co2data = co2data.set_index('Year')

co2data1 = co2data[70:]

#plt.figure(figsize=(12,4))
#sns.lineplot(x='Year', y= 'CO2', data = co2data1)

# loading the trained model
pickle_in = open('model_arima_712.pkl', 'rb') 
model_arima = pickle.load(pickle_in)

pickle_in = open('HW_model.pkl', 'rb') 
model_HW = pickle.load(pickle_in)

train = co2data1[:101]
test = co2data1[101:]
st.title("Given CO2 Historical Data")
st.line_chart(co2data1)
st.sidebar.title("CO2 Forecasting ML App")
option = st.sidebar.selectbox('Select Forecast period',
     ('5 years', '10 years', '15 years', '20 years'))
    
if option == '5 years':
        f_period = 5 
if option == '10 years':
        f_period = 10 
if option == '15 years':
        f_period = 15 
if option == '20 years':
        f_period = 20 


df1 = co2data1
from pandas.tseries.offsets import DateOffset
future_dates=[df1.index[-1]+ DateOffset(years=x)for x in range(0,f_period)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df1.columns)
future_datest_df.tail()
future_df_ARIMA=pd.concat([df1,future_datest_df])
future_df_HW = pd.concat([df1,future_datest_df])
future_df_SARIMA = pd.concat([df1,future_datest_df])


def main():       
   
      
    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
    
    if st.sidebar.button("ARIMA Model"):
        future_df_ARIMA['forecast_ARIMA']= model_arima.predict(start = future_df_ARIMA.index[145], end = 145 + f_period)   
        fpred = pd.DataFrame(future_df_ARIMA["forecast_ARIMA"][145:])
        st.title("Forecasted CO2 Data by ARIMA Model")
        st.dataframe(fpred)
        st.line_chart(future_df_ARIMA)
        
    if st.sidebar.button("Holt-Winter Model"):
        future_df_HW['forecast_HW']= model_HW.predict(start = future_df_HW.index[145], end = 145 + f_period)   
        fpred = pd.DataFrame(future_df_HW["forecast_HW"][145:])
        st.title("Forecasted CO2 Data by Holt_Winter Model")
        st.dataframe(fpred)
        st.line_chart(future_df_HW)
        
    
    
    
if __name__=='__main__': 
    main()


