#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
 
HtmlFiletime = open("PredTime.html", 'r', encoding='utf-8')
pred = HtmlFiletime.read()
    
st.set_page_config(layout="wide")
st.title('MADRID AIR POLLUTION PREDICTION')
"This is the web application where to show the result of the TFM done in KSchool during the DataScientist Master 2sd streaming edition 2021-2022. "
"All the code can be found in a public git repository: git@github.com:corti-acs/KS_TFM_MadridAirQuality.git."


st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
"Prediction TimeStamp: " + pred
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

col1, col2, col3 = st.columns((3,4,1))

with col1:
    source1 = Image.open("LineEvo.png")
    col1.subheader("NO2 MEASUREMENT EVOLUTION")
    "                                                "
    "                                                "
    col1.image(source1, use_column_width=True)
    "Evolution of N02 measurement today"  
    source2 = Image.open("BoxPlot.png")
    col1.image(source2, use_column_width=True)
    "N02 value distribution per Air Station"
    
with col2:
    #showing folium map html
    HtmlFile1 = open("MadridMap.html", 'r', encoding='utf-8')
    asmap = HtmlFile1.read() 
    col2.subheader("AIR STATION OVERVIEW")
    st.text("(Click on the marker of every station to get the prediction chart)")
    components.html(asmap,height = 7500)
    

    "Notes about predicted model used:"
    "This prediction has been done based on a SARIMA model, with the following metric accuracy"

with col3:
    col3.subheader("Legend")


# In[ ]:




