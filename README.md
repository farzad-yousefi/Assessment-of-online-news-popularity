# Assessment-of-online-news-popularity

## Introduction:
Predicting popularity in the Internet is a challenging and non-trivial task due to a multitude of factors impacting the distribution of the information: external context, social network of the publishing party, relevance of the video to the final user, etc. In this project, I have tried to use LSTM RNN to predict the popularity of online articles from mashable.com by using only the title of the articles.

## CSV file description:
There is a csv file in this repo: [OnlineNewsPopularity.csv](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/OnlineNewsPopularity.csv.)

This file was downloaded from [UCI Machine Learning Repository.](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) The first column in this file has the urls of articles:
```python
df = pd.read_csv('OnlineNewsPopularity.csv')
```

## Data Scraping:
The 'url' column in this file was used to scrape Title and content of almost 40,000 articles from mashable.com and dumped to a Mongodb for future parsing.
```python
client = MongoClient('localhost', 27017)
db = client.capstone
articles = db.articles

for url in df['url']:
    r = requests.get(url)
    try:
        articles.insert_one({'html': r.content})
    except:
        pass
```






