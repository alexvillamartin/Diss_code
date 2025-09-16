# Outline of Code

## Forecasting

XXXYYY Forecasting Results currency files show MSE, MAE, Mincer Zarnowitz and OLS regression results across all horizon. 

estimation_forecast_functions_var include MLE estimation and recursion forecasting for all volatility models considered. We explicitly mdoel and forecast variance here as required by the theory. 

These are two key files used in thesis. Other forecasting files such as 'Forecasting of Models VX' or the SVM forecasting are files from general research/testing of models. 

## Strategy

The EURUSD strat new xxx file is the final strategy results for EURUSD with the vega pnl analysis used in the thesis. Unfortunately I couldnt change name after adding so it has some extra name parts. 

Similarly we have results for other currencies and these have vega output but was not completed as not needed.  

In terms of the actual code, I attached all hedging strategy codes I used but main one of focus is 'hedging_strategy_class_NEW_KURT_SKEW_vega_VAR' as this has variances used for strategy to align with variance forecasts + includes OLS correction code integrated. 

## Modelling Smile 

Only one file here and it is where we compute the skew and kurt needed for OLS features. 

## Multifractal Analysis

'Empirical_data_MF-DFA' - results for absolute returns used in final thesis. 

'MFDFA_classes' - code that does all analysis for simulated and empirical. Includes cleaning needed to analyse squared/absolute. 

'MFDFA_functions' - my original analysis functions I used at the start of research. 

'MFDFA_new' - efficient MFDFA algo from paper "https://arxiv.org/abs/2104.10470". 

'iaaft' - algo for surrogat generation. Copy is literally the same. 

'Empirical Data Analysis New' - original analysis I did on all raw/absolute/squared returns. 

'Empirical Data Analysis' - Original analysis done on daily + secondly data with the properties of empirical time series also here. 

'joint spectrum NEW' - analysis done on the simulated data. 

'p_value_analysis' - results for my empirical / simulated p values of the surrogate hypothesis tests

'surr_plots' - one loop through iaaft algo plot

## Simulations of Models

All the 'spot (NEW) under xxx' is from the original simulated models we did. Involves spot, return, vol data and also exmaines the properties of the simulated data. 

The variety of MSM params does the MFDFA analysis on variety of different param values. 

## Other

'simple_strat_funcs' - includes cleaning of the implied vol data used in FINAL results but also has old strategy code with long/short (not as clean or updated as hedging one I refined)

