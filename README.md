# Stock Price Prediction

## Description

Time series forecasting in python leveraging machine learning concepts from the sktime library for financial analysis

This project aims to evaluate multiple time series forecasting models to predict daily closing stock price of one company listed on Nasdaq (IBM), applying classic statistical methods as well as machine learning tools and concepts.

The goal is to identify model(s) that balance prediction accuracy and interpretability for application in real-life business use-cases by financial analysts.

The project's main goal is to gain understanding of diverse forecasting models leveraging machine learning techniques and tools in python and the sktime library.

i) Explore classical statistical forecasting models

ii) Optimization applying of machine learning techniques and tools in python

iii) Assess value-added of using these forecasting models for Financial Analysis


## Installation

Clone the repository.
Change into the project directory.
(Recommended) Set up a virtual environment.
Install the required packages using requirements.txt.

## Contributing

Community contributions are welcome.

 
## Scope and Focus
 
Exploring multiple company stock price data to select one of them to predict future prices for the following period (90 days)
 
## Data Overview
The dataset used for this project consists of daily stock closing prices from 2014 to 2023 of selected companies listed on Nasdaq, focussing on IBM for forecasting. The data transformation technique employed includes differencing to handle non-stationarity in the time series data.

The data consists of daily closing stock prices from '01 Jan 2013'
until '31 Dec 2023' for the following Indexes and companies:

 - ^GSPC: The S&P 500 Index measures the stock performance of 500 large companies listed on stock exchanges in the United States. It is considered an indicator of the overall health of the U.S. equity market 12.
- ^DJI: The Dow Jones Industrial Average (DJIA) tracks the stock performance of 30 of the largest American corporations in the private sector. It is often referred to as the "Dow" or simply "the Dow" 1.
- ^IXIC: The NASDAQ Composite is a market-weighted index of all common stocks traded on the NASDAQ Stock Market, including the National Market System. It includes approximately 3,000 securities 1.
- ^NDX: The Nasdaq-100 Index is a stock market index composed of 100 of the largest domestic and international non-financial companies listed on the Nasdaq Stock Market 1.
- AAPL: Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells smartphones, personal computers, tablets, wearables, and accessories, and sells a variety of related services 1.
- GOOGL: Alphabet Inc. is an American multinational conglomerate specializing in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware 1.
- MSFT: Microsoft Corporation is a technology company that develops, manufactures, licenses, supports, and sells computer software, consumer electronics, personal computers, and related services 1.
- IBM: International Business Machines Corporation (IBM) is a multinational technology company headquartered in Armonk, New York, United States, with operations in over 170 countries 1.
- NVDA: NVIDIA Corporation is a world leader in artificial intelligence computing. It specializes in GPU accelerated computing technology and software solutions 1.
- META: Meta Platforms Inc., formerly known as Facebook Inc., is a social networking service and digital media company based in Menlo Park, California. It owns several popular internet properties, including the Facebook social network, Instagram photo-sharing app, and WhatsApp messaging service 1.

## Methodology

Programming language: python. Main Libraries used: pandas, numpy, sktime, statsmodels, yfinance

- Obtaining stock price data from yahoo finance through python's yfinance library

- Initial EDA to understand the general trends of main indexes (S&P, Dow Jones, Nasdaq 100 and how similar the price curves of the tech companies in the data was)

- Selection of company to use for forecasting (IBM): due to the unique nature of IBM's price development curve over the selected time period (10Y) IBM has been selected as it did not follow the same trends as Nasdaq 100, Apple Microsoft (whch show a long term growth curve that increases in teh last 3-4 years), but instead it showed an initial downward trend, then relative price stability, continuing then to show the upward trend as the other companies have shown in the last 3-4 years.

- Data Cleaning: fixing / format / dates, missing data.

- Data Analysis to assess whether it can be used for time series forecasting or if it needs transformation:
	- Preparing visualization to analyze the data
	- Testing for starionarity using ADF and KPSS tests
	- Autocorrelation, and partial autocorrelation
	
- Train/Test split using temporal window

- Data transformation: differencing
		- 2nd round of data analysis post-transformation

- Model training, predictions, and evaluation. Model parameter determination techniques:
	- using previous statistical analysis of data to determine the models' parameters (e.g. ARIMA)
	-  Gridsearch and Randomized Gridsearch with Cross Validation, using data sorted training data, split into Expanding and/or Sliding Windows for other models
	- Using only pre-set settings (e.g. Prophet)

- Evaluating the prediction:
	- test period prediction and comparison to y_true (scores used= RMSE, MAE, MAPE)

- Model Selection based on y_pred vs y_true scores for the y_test period and considerations regarding the intrinsic characteristics of each model which impacts their use positiblies

- Cross Validation of selected models, across expanding windows (scores = RMSE. please note resulting df displays 'MSE' in the column name although it's calculation is 'RMSE')

## Models and Algorithms

Models considered for time series forecasting using the time series stock price data:
- Simple Moving Average (SMA)
- Simple Exponential Smoothing (SES)
- AutoRegressive Integrated Moving Average (ARIMA), and AutoARIMA
- Prophet

Models considered using tabular data (method: sktime's make_reduction)

- GradientBoosting Regressor
- KNN Regressor

## Results and Findings


### Data Exploration Phase: 

The initial data exploration suggests that the Nasdaq top 100 (^NDX) naturally reflects its main components (strong tech bias), so its growth curve is similar to the ones from Apple, Microsoft, Meta, wheras IBM, and NVIDIA showed different trends over the last 10 years.

All models were tested with transformed as well as not transformed data, in some cases using multiple parameters to identify the best ones, wither manualy/visually usign for loops and manual adjustments, automatically using gridsearchcv and randomizedgridsearchcv.

Limitatons of this analysis:
computing power limited the use of more advanced testing techniques that were reserved for the selected models in round 2.
Residuals were not assessed, only visualized for documentation purposes.

Based on initial findings from the model error scores from the y_test period, and model characteristics, the following assessment was made:

### Model Comparison round 1:

forecasting methods based on time series forecasting::

- Simple Moving Average (SMA): due to its simplicity it could be beneficial, but will be excluded from teh selected models due to low flexibility and low accuracy.

- Simple Exponential Smoothing (SES): this model represents in this analyst view the benchmark to compare other, seemingly more advanced models. It's accuracy tests were satisfactory and it's interpretability for a financial analyst and others is quite straighforward. It was selected for the final test round.

- AutoRegressive Integrated Moving Average (ARIMA): this model is highly configurable, this characteristic makes it more time consuming to use efficiently, as determinig the model parameters requires a basic undestanding of statistical concepts, which also affects it's interpretability for other potential stakeholders.

- Prophet: this model was highly accurate in the first testing round, even though separate gridsearch and cross validation had been executed. The results are somewhat a black-box unless the analyst can dedicate time to understanding the methodology, without which usage of any model is discouraged. It was selected for the final test round.

Machine learning regressors based on tabular data (method: sktime's make_reduction)

- GradientBoosting Regressor: not considered due to lower scores than KNN regressor

- KNN Regressor: satisfactory results made this model, even though it was not a time sereis, a strong contender. It was selected for the final test round to assess whether after cross validation it can be expected perform as well as the other models.

### Model Comparison round 2:

- Model 1. t(1) simple exponential smoothing (SES) * gridsearch 
- Model 2 t(0) Prophet
- Model 3. t(0) ARIMA
- Model 4. Combi Model: combined model of SES & Prophet
- Model 5. t(1) KNN Regressor:

## Future Work

increasing the list of company stocks analyzed to an hierarchical panel (stock index + multiple companies from a specific sector, e.g. Tech.) with stock clustering, and global model training to capture cross-market relationships and improve the overall accuracy of the forecasts.

Use-case analysis for different financial analysis tasks as well application of these forecasting techniques to different organization types

advanced ensemble methods to optimize predictions

## Acknowledgments

Selected bibliography: "Introductory Econometrics for Finance" by Chris Brooks. Cambridge University Press, 2019.

"Statistical forecasting: notes on regression and time series analysis" by Robert Nau, Fuqua School of Business, Duke University. https://people.duke.edu/~rnau/411home.htm

Python Library: The SKTime Python library. (https://www.sktime.net/en/stable/index.html)

​Websites: Medium.com. TowardsDataScience. Analytics Vidhya. Rob J Hyndman. (Accessed: January 26, 2024). 

Project guidance:
I. Soteras, S. Atawane.