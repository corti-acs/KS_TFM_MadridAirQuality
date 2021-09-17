# KS_TFM_MadridAirQuality
This repository contains Study to predict pollution peaks in Madrid city as part of a TFM for 2sd edition of Data Scientist Master Streaming 2021-2022 by KSCHOOL


# Business Problems

The objective of this study is to elaborate a method that can predict pollution peaks in Madrid city, identifying variables impacting on NO2 index
•	WHY.  
•	Pollutions peaks in Madrid are triggering traffic restrictions that can have an impact on the normal traffic flow in the city, disturbing business and citizens activities.
•	In addition pollutions peaks have an impact on the health of the weakest or most vulnerable Madrid citizens,  so being capable to predict these episodes government could send recommendations to vulnerable population, trying to prevent the impact.
•	HOW. Creation of a prediction model capable to predict episodes of max pollution(based on NO2 index) 2 days in advance applying different machine learning (ML) techniques, calculating the probability to reach high pollution episodes
•	BONUS. 
•	Analysis of the impact of traffic restrictions in the traffic congestion index.
•	Analysis of pollution impact in the citizens live per district.

# Project Organization
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- General documentation & Memo
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
   ├── __init__.py    <- Makes src a Python module
   │
   ├── data           <- Scripts to download or generate data
   │   └── make_dataset.py
   │
   ├── features       <- Scripts to turn raw data into features for modeling
   │   └── build_features.py
   │
   ├── models         <- Scripts to train models and then use trained models to make
   │   │                 predictions
   │   ├── predict_model.py
   │   └── train_model.py
   │
   └── visualization  <- Scripts to create exploratory and results oriented visualizations
       └── visualize.py

![image](https://user-images.githubusercontent.com/82566952/133854158-f1e94a36-d49e-4224-b182-2e0336ce308b.png)
