This repository contains code for traffic matrix (TM) prediction. 
- Abilene data can be downloaded from: https://www.cs.utexas.edu/~yzhang/research/AbileneTM/
- GEANT data can be downloaded from: https://sndlib.put.poznan.pl/home.action

Further details provided in our work:  

Each folder contains the following files: 
- mlu_baseline.py: run the MLU baseline on the original dataset
- predict_OD_mse.py: train N^2 GRU models to predict each OD pair in the TM
- predict_TM_cosine.py: train one GRU model to perform entire matrix prediction via cosine similarity loss function
- predict_TM_mse.py: train one GRU model to perform entire matrix prediction via MSE loss function
- predict_TM_denoise.ipynb: perform denoise pre-processing step on TMs and do prediction as in predict_TM_mse.py

Required packages:
- numpy, pytorch, gurobipy, sklearn, matplotlib
