# ClimatePredictionChallengesProject3
<span style="color:red">Team Member: Mujahed Darwaza, Bowen Han, George Lu, Arnav Saxena

This research project focuses on studying feature importance in the ML model that reconstructs surface ocean pCO2. Our main concern was to analyze and validate the physical nature of the model, and identify whether additional data was required to improve it. We conduct a SHAP value analysis to validate that trends in the features make physical sense. We also investigate connections between model inaccuracies and their respective features.\
Parts 1 and 2 of the code set up the workspace and create the reconstruction with member r1r10 from the CanESM model.\
(https://colab.research.google.com/drive/1EQDxB9K5RVe_TYypgkX3EEgA5ZbZFeDL?usp=sharing) \
Packages need to install:
~~~python
!pip install SkillMetrics
!pip install cmocean
!pip3 install cartopy
!pip uninstall -y shapely
!pip install shapely --no-binary shapely
!pip install shap
~~~
Packages need to import: 
~~~python
#@title import necessary packages
import os
import datetime
from pathlib import Path
from collections import defaultdict
import scipy
import random
import numpy as np
import xarray as xr
import pandas as pd
import joblib
import pickle
import skill_metrics as sm
import math
import seaborn as sns
import cmocean as cm            # really nice colorbars
import matplotlib.pyplot as plt # for making plots
import shap
from sklearn.model_selection import train_test_split
~~~
## Part 1: SHAPLEY values
What are shapley values?\
Shapley values is a concept of cooperative game theory where different players work together (in cooperation) towards a common goal and are awarded payouts inline with their individual contributions. In our case the goal is the prediction for a particular instance, the players are all the feature values of an instance, and the gain is the actual predictions for this instance minus the average predictions for all the instances.
To be put simply, shapley values indicate how much a feature value contribute to the prediction. To be more precise, shapley value is the average marginal contribution of a feature value across all possible coalitions.

Please refer the following for an in-depth explaination:
https://christophm.github.io/interpretable-ml-book/shapley.html \
https://www.youtube.com/watch?v=NBg7YirBTN8&t=11s 
## 1.1 Display shap values for XGBoost
![image](https://user-images.githubusercontent.com/59796732/169884352-30e9e456-57cc-44c6-abe6-3e8401334d32.png) \
Feature Importance (Figure 1): Here we can see that the heaviest feature dependence is on the variable A, which is just a function of latitude. Since latitude doesn't change over time, we will ignore it in our temporal analysis.\
![image](https://user-images.githubusercontent.com/59796732/169885102-e8b2ee0c-bb84-4cd4-9474-9bd452941fe5.png) \
Swarm Plot(figure 2): Showcases how exactly the shapley value contribution varies with the feature values. For instance higher values of xCO2 have a positive contribution on the prediction values while lower xCO2 values have a negative contribution on the prediction values.
## 1.2 Dependence Plots
![image](https://user-images.githubusercontent.com/59796732/169886064-712c4a70-7102-4b09-9dcd-ace206d22db7.png) \
Interaction plots between the most important features: These plots show how the shapley value impact of a variable is affected by other variables. Let's look at the 2nd plot (xCO2 vs mld_clim_log) clearly, for instances with lower xCO2 values and lower mld_clim_log(blue) have lower shapley values as compared to those with lower XCO2 but higher mld_clim_log values. However, this trend reverses for higher xCO2 values.
## Part 2: Examining seasonal feature values and their importance in different regions
We have define different regions (South Sub-tropical, North Sub-tropical, Tropical and Pole area) and try to look at the trend.\
![image](https://user-images.githubusercontent.com/59796732/169887481-98257adc-910a-40ee-a821-f840ed1182df.png)
Above, we can identify some physically sensible trends. We see the strong seasonalities in temperature (and also mixed layer depth) in the north and south subtropics, as expected. We also notice that temperature and xCO2 variables have a direct relationship with how important it is in the model. I.e., the further a feature value deviates from average, the stronger of an impact it has on the model in the same direction. This makes sense for the variables xCO2 (dry air mixing ratio of atmospheric CO2) and SST since the higher mixing ratio would imply more pCO2 in the water, and in the subtropics, the pCO2 content tracks the temperature (Takahashi et al., 1993). \
On the other hand, we can see that the chlorophyll anomaly (chl_anom) and the mixed layer depth (mld_clim_log) feature values have almost an inverse relationship with the SHAP values. This makes intuitive sense as biological activity would reduce pCO2 in the water (by taking it in and capturing it as calcium carbonate) and a deeper mixed layer would mix the surface pCO2 deeper into the water, which reduces the amount at the surface. \
An interesting takeaway is how the seasonalities become less apparent in the tropics, which again makes sense because there is simply less seasonality there. However, we see in the next plot below (from the starter code, linked here) that this is a region where the model has some inaccuracies as well. Therefore, it is possible that the lack of seasonality accuracy in the tropics for the model arises simply due to a lack of seasonality of the features themselves.in all these regions. 
## Part 3: Shap values per region 
Tropical: \
![image](https://user-images.githubusercontent.com/59796732/169888587-216ad0ae-8e65-408c-a148-97a926dca318.png) \
South Sub-tropical:\
![image](https://user-images.githubusercontent.com/59796732/169888755-450cbfb9-0765-4ceb-99a6-00b4e84abda5.png) \
North Sub-tropical:\
![image](https://user-images.githubusercontent.com/59796732/169888899-32f5a4ab-b41c-4450-8602-5768d44169b9.png) \
We wanted to investifgate further the reason why seasonal discrepancy was showing biases in certain regions. We decided to focus on the areas where there is the least seasonal correlation : the tropics and the south. The tropics, as explained earlier, have probably less seasonal correlation due to the physical system itself not showing high seasonality. \
Nevertheless, there seems to be a discrepancy in the south sub-tropical region as well. Interestingly, if we refer back to our seasonal plots, we see that one of the most important variables in the south is the chlorophyll anomaly. This variable shows a seasonality both in the north and south, but only the north shap values match that seasonality. This is visualized further if we look at spatial plots. \
![image](https://user-images.githubusercontent.com/59796732/169889091-12e20fa9-d9d1-440e-b8dc-c08350403b50.png) \
There is a clear band of high chl_anom (higher biological production) in the south. This band matches with the region where seaonality is not accurately reflected in our initial plot. Therefore, it is possible that the reconstruction fails to accomodate the seasonality of biological cycles in the southern ocean, and thus result in a failure to accurately capture seasonality of pco2 in its reconstruction. \ 
## Part 4: Discussion \
We applied the same methodology to other members of the XGBoost model as well as random forest and LightGBM, and they have achieved similar results(ie: same top five features). This is available in the directory under "different models" or "new members". \
In conclusion, our SHAP value analysis has helped us identify the most important features to consider when reconstructing pCO2 values from the CanESM model. With this newly acquired information, we can prioritize the data that requires more precise modelling or to be collected for a better reconstruction of pCO2. Furthermore, the highlighted features validate the use of the Machine Learning models : they capture physical relationships between features that make sense. Finally, analyzing the trends in our features through the SHAP value method allows one to identify the potential failure points in the reconstruction. More specifically, we have showed a potential failure in capturing the importance of the biological activity in the southern ocean, which might explain the poor estimation of seasonality in pCO2. \
