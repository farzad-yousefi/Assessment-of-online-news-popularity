# Assessment of Online News Popularity

## Introduction:
Predicting popularity in the Internet is a challenging and non-trivial task due to a multitude of factors impacting the distribution of the information: external context, social network of the publishing party, relevance of the video to the final user, etc. In this project, I have tried to use LSTM RNN to predict the popularity of online articles from mashable.com by using only the title of the articles.

## CSV File Description:
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
By defining and using two functions (to get the url and article title), the results were parsed and written to another CSV  file [article_titles_urls.csv](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/article_titles_urls.csv).  
```python
def soup(n):
    return BeautifulSoup(articles.find()[n]['html']).find("header", {"class": "article-header"}).find("h1").text
    
def soup3(n):
    return BeautifulSoup(articles.find()[n]['html']).find("header", {"class": "article-header"}).find("h1")['href']

f = open("article_titles_urls.csv", "a")
for i in range(39645):
    try:
        f.write(soup3(i)+","+soup(i)+","+'\n')
    except:
        pass
    if i%1000 == 0:
        print(i)
f.close()
```
You can find the complete notebook for this part [here.](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/Scraping_parsing_article_titles.ipynb)

## Merging Article Titles with the Original CSV File
Here, I have used the url as a key to merge titles to their associated row in the original CSV file. My goal here, is to use the 'number of shares' column in the [original CSV file](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/OnlineNewsPopularity.csv) and 'titles' column in [the scraped article titles.](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/article_titles_urls.csv)




