# Thai stock movement prediction using Graph Neural Network
This repository contains the source code for "Thai stock movement prediction using Graph Neural Network" - the senior project in Special Topics in Insurance 2603494. In this source code, we implemented three types of model: Hierarchical Graph Neural Network ([refers to](https://arxiv.org/abs/1908.07999)), Long-short Term Memory, and Multilayer Perceptron. In short, the concept of HATs is to utilize the relationship between stocks to get the better representation for each node (stock). 
<br> </br>

## Dataset
We select the initial dataset of stocks based on SET100 and the other stocks that have a certain relation with stocks in SET100. We scraped, in total, 778 stocks using yahoo-finance api in Python ([refers to](https://pypi.org/project/yfinance/)). Our dataset, both price-related data and relation data, is available in this repository in the **data** directory.

## Usage
To replicate the result in the paper, run **main.py** file and change the value of each hyperparameter in **config.py** file. However, all of the initial value in the file are the configuration that we used in paper.

## Result
Result from this study can be found in this google drive ([link](https://drive.google.com/drive/folders/1Gbw4Q2ABm7DMn2MjX5NnNvlO6miWFdgk?usp=sharing))
