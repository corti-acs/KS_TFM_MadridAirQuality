#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
 

#Streamlit dashboard layout    
st.set_page_config(layout="wide")
st.title('MADRID AIR POLLUTION PREDICTION')
"This is the web application where to show the result of the TFM done in KSchool during the DataScientist Master 2sd streaming edition 2021-2022. "
"All the code can be found in a public git repository: git@github.com:corti-acs/KS_TFM_MadridAirQuality.git."

HtmlFiletime = open("PredTime.html", 'r', encoding='utf-8')
predtime = HtmlFiletime.read()
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
"Prediction TimeStamp: " + predtime
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

col1, col2, col3 = st.columns((.7,1.7,1.5))

#folium map legend
with col1:
    
    col1.subheader("Legend")
    source1 = Image.open("MapLegend.png")
    col1.image(source1, use_column_width=True)

#folium map
with col2:
    
    HtmlFile1 = open("MadridMap.html", 'r', encoding='utf-8')
    asmap = HtmlFile1.read() 
    col2.subheader("AIR STATION OVERVIEW")
    st.text("(Click on the marker of every station to get the prediction chart)")
    components.html(asmap,height = 650)
    "Notes about predicted model used:"
    "This prediction has been done based on a SARIMA model, getting the prediction accuracy"
    source2 = Image.open("Metrics.png")
    col2.image(source2, use_column_width=False)    

#charts
with col3:
    col3.subheader("NO2 MEASUREMENT EVOLUTION TODAY")
    "   "
    "   "
    source3 = Image.open("BoxPlot.png")
    col3.image(source3, use_column_width=True)
    source4 = Image.open("LineEvo.png")
    col3.image(source4, use_column_width=True)


# In[ ]:




