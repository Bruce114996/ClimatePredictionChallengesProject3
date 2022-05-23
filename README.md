# ClimatePredictionChallengesProject3
<span style="color:red">Team Member: Mujahed Darwaza, Bowen Han, George Lu, Arnav Saxena

This research project focuses on studying feature importance in the ML model that reconstructs surface ocean pCO2. Our main concern was to analyze and validate the physical nature of the model, and identify whether additional data was required to improve it. We conduct a SHAP value analysis to validate that trends in the features make physical sense. We also investigate connections between model inaccuracies and their respective features.\
Parts 1 and 2 of the code set up the workspace and create the reconstruction with member r1r10 from the CanESM model.\
(https://colab.research.google.com/drive/1EQDxB9K5RVe_TYypgkX3EEgA5ZbZFeDL?usp=sharing)\
Packages need to install 
~~~python
!pip install SkillMetrics
!pip install cmocean
!pip3 install cartopy
!pip uninstall -y shapely
!pip install shapely --no-binary shapely
!pip install shap
~~~
Packages need to import 
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
