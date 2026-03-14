# Solution-1

## Overview

The goal is to understand how features like RAM, storage, camera, battery capacity, and brand affect pricing.

## Dataset

The dataset contains information about different smartphones including:

* Brand
* RAM (GB)
* Storage (GB)
* Camera (MP)
* Battery Capacity (mAh)
* 5G Support
* Price (INR)

### 1. Data Collection

Smartphone specifications were collected from e-commerce product catalogs.

### 2. Data Cleaning

Steps performed:

* Removed duplicate rows
* Checked missing values
* Encoded categorical variables
* Verified numeric columns

### 3. Exploratory Data Analysis (EDA)

Key analyses performed:

* Correlation heatmap
* Price distribution
* RAM vs Price relationship
* Storage vs Price relationship
* Battery capacity vs Price

### 4. Visualization

Libraries used:

* Matplotlib
* Seaborn

Plots created:

* Scatter plots
* Distribution plots
* Correlation heatmap

### 5. Model Building

A **Linear Regression model** was used to predict smartphone prices.

Steps:

1. Feature selection
2. Train-test split
3. Model training
4. Performance evaluation

Evaluation metrics:

* R² Score
* Mean Squared Error


## Results

Key predictors of smartphone price:

* Brand reputation
* RAM size
* Storage capacity
* Camera resolution
* 5G support

These features significantly influence smartphone pricing in the market.

