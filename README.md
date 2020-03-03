# Assessment-of-online-news-popularity

## Introduction:
Predicting popularity in the Internet is a challenging and non-trivial task due to a multitude of factors impacting the distribution of the information: external context, social network of the publishing party, relevance of the video to the final user, etc. In this project, I have tried to use LSTM RNN to predict the popularity of online articles from mashable.com

## CSV file description:
There is a csv file in this repo: [OnlineNewsPopularity.csv](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/OnlineNewsPopularity.csv)

This file was downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) The first column in this file has the urls of articles:
```python
df = pd.read_csv('OnlineNewsPopularity.csv')
```

## Data Scraping:
Title and content of almost 40,000 articles were scraped from mashable.com and dumped to a Mongodb for future parsing.
```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```






