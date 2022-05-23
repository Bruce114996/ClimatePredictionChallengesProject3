# ClimatePredictionChallengesProject3
<span style="color:red">Team Member: Mujahed Darwaza, Bowen Han, George Lu, Arnav Saxena

This research project focuses on studying feature importance in the ML model that reconstructs surface ocean pCO2. Our main concern was to analyze and validate the physical nature of the model, and identify whether additional data was required to improve it. We conduct a SHAP value analysis to validate that trends in the features make physical sense. We also investigate connections between model inaccuracies and their respective features.\
Parts 1 and 2 of the code set up the workspace and create the reconstruction with member r1r10 from the CanESM model.\
(https://colab.research.google.com/drive/1EQDxB9K5RVe_TYypgkX3EEgA5ZbZFeDL?usp=sharing)
~~~python
!pip install SkillMetrics
!pip install cmocean
!pip3 install cartopy
!pip uninstall -y shapely
!pip install shapely --no-binary shapely
!pip install shap
~~~

