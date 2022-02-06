# KSchool TFM MadridAirQuality
This repository contains Study to predict pollution in Madrid city as part of a TFM for 2sd edition of Data Scientist Master Streaming 2021-2022 by KSCHOOL


# ABSTRACT

Pollution around the main urban agglomerations across the globe is one of the main problems that governments of all countries try to mitigate every year. 

This situation was growing and growing during last centuries, reaching the current levels that are causing the death of more than 7 Mill of inhabitants per year (data from Paris climate conference  . WHO); a part of impacting in the right child development, increasing the risk of suffering from acute respiratory diseases, and developing chronic diseases. 

In addition, the pollution has been identified as one of the main reasons of climate change.

Governments from the main countries are trying to take decisions to reduce it… but without a relevant success, just only agreeing some “cosmetics” actions that are not attacking the problem from the root, probably due to the complexity to balance those measurements together with the economic country development.

WHY
One of the tools available for the governments to mitigate and prevent is the monitoring of the main pollution parameters in the air and try to estimate them for the future to trigger preventive actions for the citizens.


WHERE
This study will be focused on Madrid, one the European cities with highest mortality in Europe due to the N02 index  .

WHAT
The goal of this study is to create a prediction model, based on the historical Air Station and Weather data network from “Ayuntamiento de Madrid”, with the objective to predict the pollution for the next 24 hours from the live data published for every Air Station.

This will be visualized in a web front-end that allow to the end user to monitor situation (almost on real-time) per AirStation and the related prediction for everyone for the next 24 hours.


Project Repository Organization
-------------------------------

 
    ├── README.md         <- The top-level README for developers using this project.
    ├── data
    │   ├── interim       <- Intermediate data that has been transformed.
    │   ├── processed     <- The final, canonical data sets for modeling.
    │   └── raw           <- The original, immutable data dump.
    │
    ├── docs               
    │   ├──GeneralDocs    <- Generated graphics and figures to be used in reporting
    │   └──MemoThesis     <- The original, immutable data dump.
    │  
    │  
    ├── models            <- Trained model. Due to GIT size limitation this is shared via dropbox
    │                        (see link in the Memo document)
    │
    ├── notebooks         <- Jupyter notebooks. 
    │
    ├── webapp            <- .py files to run to get live data prediction and to launch streamlit dashboard. 
    │                         Also include different files used in this dashboard
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
   


    

