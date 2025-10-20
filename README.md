Spatio-Temporal Graph Neural Network and Kriging Interpolation Comparison

This project implements and compares two spatial interpolation methods on meteorological data:

Spatio-temporal graph neural network (ST-GNN)
Kriging interpolation
The models are used to predict annual average temperature at weather stations in China based on real observational data.

Files

ST-GNN.py: A spatio-temporal graph neural network based on GCN

Enhanced-Graph-Transformer.py: An improved model using graph attention and Transformer architecture

The meteorological data used in this study is processed from the Global Summary of the Day (GSOD) dataset, which is publicly available at:

https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/

The processed dataset (yearavetemp.csv) includes station coordinates (latitude and longitude), elevation, and annual mean temperature.

Data Description

The file yearavetemp.csv contains the following fields:

Year: observation year

Station ID: unique identifier for each weather station

Longitude: geographic longitude

Latitude: geographic latitude

Elevation: altitude in meters

Annual Mean Temperature: average annual temperature in degrees Celsius

Data source: Publicly available meteorological observations from the China Meteorological Administration (anonymized)

Usage

Place yearavetemp.csv in the project root directory
Run ST-GNN.py or Enhanced-Graph-Transformer.py
The script will train the model and output evaluation results such as RMSE
Notes

This code is intended for academic research and educational purposes only
Do not use for commercial applications or public distribution without permission
