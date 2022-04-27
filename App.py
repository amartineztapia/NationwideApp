# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:23 2022

@author: alber
"""
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
DNN_Location = 'C:/Users/alber/Documents/Spring 22/Section 2/Practicum/Data/Deep Neural Networks/'
os.chdir(DNN_Location)
from Window import *
from tensorflow import keras
import tensorflow as tf
from datetime import timedelta

df0 = pd.read_csv(DNN_Location + 'data.csv')
st.set_page_config(layout = "wide")

from PIL import Image
image = Image.open('C:/Users/alber/Documents/Spring 22/Section 2/Practicum/App/Nationwide.png')

def header(cont):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{cont}</p>', unsafe_allow_html=True)


st.image(image, width=256)

st.title('Credit Spreads Wizard')
st.subheader('Last updated on April 26th, 2022')

date = list(df0['Date'])
dfindicators = list(df.columns)
col1,pad1,col2 = st.columns([10,1,10])
with col1:
     st.markdown('''
          Select a set of G7 countries from the list on the left - you can add or delete as many as you want - 
          and then adjust the left slider to the first quarter of interest and the right one to last quarter 
          that you want data for.
          The first quarter that there is data for is Q! 2007 and the final one is Q4 2021.
          When you have selected the countries and the time period should select a data table.
          The tables available are:
          - __GPSA__ - Growth rate based on seasonally adjusted volume data, percentage change from previous quarter
          - __GYSA__ - Growth rate based on seasonally adjusted volume data,  percentage change from same quarter of previous year, 
          - __VIXOBSA__ - Volume index, OECD reference year, seasonally adjusted
          ''')
with col2:

     selected_indicators = st.multiselect(
          'Select Indicators',
          dfindicators, dfindicators[:5])

     start_period, end_period = st.select_slider(
          'Select a range ',
          options=date,
          value=(date[-130],date[-1]))

     startindex = date.index(start_period)
     endindex = date.index(end_period)

fig, ax = plt.subplots()
df[startindex:endindex][selected_indicators].plot(figsize=(12,4),ax=ax)
ax.legend(loc=3)
st.pyplot(fig)


## Our predictions

  # Window that takes multiple steps as inputs
LABEL_WIDTH = 23 
SHIFT = 130                                                  # Has to be the same as CONV_WIDTH


## Plot


##############################################################################
##############################################################################

col3,pad2,col4 = st.columns([10,1,10])

with col3:
    st.title('A Very Useful Widget')
    selected_indicators2 = st.multiselect(
          'Select Indicators',
          dfindicators, dfindicators[:5], key = "<selected indicators>")
    single_ind = st.selectbox("Select the indicator that you would like to alter below", options = (['<select>'] + selected_indicators2), key = "<single ind>")
    if single_ind != '<select>':
        change_ind = st.selectbox("Select what would you like to modify it", options = ('<select>', 'Volatility', 'Slope'), key = "<change ind>")
        if change_ind == 'Volatility':
            std_mult = st.select_slider('Std Multiplier', options=[.25,.5,1,2,3], key = "<std mult>", index = 2)
        if change_ind == 'Slope':
            slope = st.select_slider('Slope', options=[-3,-2,-1,0,1,2,3], key = "<slope>", index = 0)
    target = st.select_slider('Select a Date', options=date, value=(date[-256]), key = "<target slider>")

date_indices2 = date_time.iloc[-130:]

dense = keras.models.load_model(DNN_Location + 'LSTM_130.h5')    # Loading the model

LABEL_WIDTH = 23 
SHIFT = 130 

df1 = pd.read_csv(DNN_Location + 'data.csv')

target_index = date.index(target)
test_range = df[(target_index-SHIFT):target_index]
test_range = (test_range - train_mean) / train_std
tensor_test = tf.convert_to_tensor(test_range)                          # Our Dependent Variable
indices = test_range.index+SHIFT
date_time2 = date_time.append(pd.Series(pd.date_range(start=date_time.iloc[-1], periods=150, freq='D', closed='right')))
date_time2 = date_time2.rename("Date", axis = 0)
date_indices = date_time2.iloc[indices]
date_indices2 = date_time2.iloc[indices-SHIFT]
a = tf.keras.utils.timeseries_dataset_from_array(np.array(tensor_test), targets = None, sequence_length = SHIFT, sequence_stride=1, batch_size=1)
o = next(iter(a))

with col4:
    
    if single_ind != '<select>':
        st.write("#")                      # Just leaving spaces
        st.write("#")
        st.write("#")
        st.subheader('The selected indicator is plotted below')
        fig3 = plt.figure(figsize=(12, 4))
        plt.scatter(date_indices2, df.iloc[date_indices.index][single_ind], c='#0071bf')
        st.pyplot(fig3)
    else:
        st.write("#")                      # Just leaving spaces
        st.write("#")
        st.markdown('''
             Select a set of G7 countries from the list on the left - you can add or delete as many as you want - 
             and then adjust the left slider to the first quarter of interest and the right one to last quarter 
             that you want data for.
             The first quarter that there is data for is Q! 2007 and the final one is Q4 2021.
             When you have selected the countries and the time period should select a data table.
             The tables available are:
             - __GPSA__ - Growth rate based on seasonally adjusted volume data, percentage change from previous quarter
             - __GYSA__ - Growth rate based on seasonally adjusted volume data,  percentage change from same quarter of previous year, 
             - __VIXOBSA__ - Volume index, OECD reference year, seasonally adjusted
             ''')

def prediction_Dense(o, datei):
    fig2 = plt.figure(figsize=(12, 4))
    scale_back = dense(o)*train_std["HY_OAS"]+train_mean["HY_OAS"]
    element = datetime.datetime.strptime("2021-06-01","%Y-%m-%d")
    if datei.iloc[-1] < element:
        actuals = df.iloc[indices]["HY_OAS"]
        plt.scatter(datei, scale_back, marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
        plt.scatter(datei, actuals, label = "Actuals")
        st.pyplot(fig2)
    else:
        fig2 = plt.figure(figsize=(12, 4))
        plt.scatter(datei, scale_back, marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
        st.pyplot(fig2)
    
if st.button("Run Predictions", key = "<Predictions>"):
    prediction_Dense(o, date_indices)

if st.button("Reset", key = "<Reset 1>"):
    prediction_Dense(o, date_indices)



fig, ax = plt.subplots()
df.iloc[date_indices.index-SHIFT][selected_indicators2].plot(figsize=(12,4),ax=ax)
ax.legend(loc=3)
st.pyplot(fig)


    
##############################################################################
##############################################################################











