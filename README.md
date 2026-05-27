# 📊 Sales Prediction using Python

## Overview
This project predicts future sales based on advertising spend across **TV, Radio, and Newspaper** channels using Python.  
It applies regression modeling to analyze how marketing investments impact sales outcomes.

## Dataset
- **File:** Advertising.csv  
- **Columns:**
  - `TV` – Advertising spend on TV  
  - `Radio` – Advertising spend on Radio  
  - `Newspaper` – Advertising spend on Newspaper  
  - `Sales` – Product sales (target variable)

## Steps
1. **Data Preparation**
   - Load and clean dataset
   - Handle missing values and drop unnecessary columns
2. **Exploratory Data Analysis**
   - Visualize relationships between advertising channels and sales
   - Identify correlations
3. **Modeling**
   - Train a **Linear Regression model**
   - Evaluate using **Mean Squared Error (MSE)** and **R² Score**
4. **Insights**
   - TV and Radio have stronger impact on sales compared to Newspaper
   - Helps optimize marketing budget allocation

## Requirements
Install dependencies before running:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
