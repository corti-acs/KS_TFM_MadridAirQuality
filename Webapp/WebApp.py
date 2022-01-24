#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
 

#Streamlit dashboard layout    
st.set_page_config(layout="wide")
st.title('MADRID AIR POLLUTION PREDICTION')
"This is the web application where to show the result of the TFM done in KSchool during the DataScientist Master 2sd streaming edition 2021-2022. "
"All the code can be found in a public git repository:"
st.write("check out this [link](https://github.com/corti-acs/KS_TFM_MadridAirQuality)")

#Reading last prediction timestamp, last time when predicted model was run
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
    
    my_expander = st.expander(label='Notes about prediction model:')
    with my_expander:
        "This prediction has been done based on a SARIMA model, using the parameters with smallest ACI, Akaike Information Criteria,(AIC is a widely used measure of a statistical model quantifying the goodness of fit and the simplicity of the model into a single figure)"
        "Also knowing that Test Dickey-Fuller proof that series are Stationary, the parameters used are: SARIMAX (3,0,1)x(3,1,1,12) "
        source5 = Image.open("model.png")
        my_expander.image(source5, use_column_width=False)    
        "Model prediction metrics are showing that current model has not a great prediction accuracy. See below values for the most common metrics used to measure prediction quality:"
        source2 = Image.open("Metrics.png")
        my_expander.image(source2, use_column_width=False)    

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




