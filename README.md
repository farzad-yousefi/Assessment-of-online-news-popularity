# Assessment of Online News Popularity

## Introduction:
Predicting popularity in the Internet is a challenging and non-trivial task due to a multitude of factors impacting the distribution of the information: external context, social network of the publishing party, relevance of the video to the final user, etc. In this project, I have tried to use LSTM RNN to predict the popularity of online articles from mashable.com by using only the title of the articles.

## CSV File Description:
There is a csv file in this repo: [OnlineNewsPopularity.csv](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/OnlineNewsPopularity.csv.)

This file was downloaded from [UCI Machine Learning Repository.](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) The first column in this file has the urls of articles:
```python
df = pd.read_csv('OnlineNewsPopularity.csv')
```
## EDA:
You can find explainatory data analysis of the CSV file [here.](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/Newspopularity_EDA.ipynb)

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

## Non-Neural Network Models
Before jumping into RNN and more advanced methods, it is reasonable to have some base models and compare their performance with advanced models. [Here](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/models.ipynb) you can find some RandomForestClassifier and GradientBoostingClassifier models with their associated ROC curves for future comparisons.
Overall, randomforest (left curve) and gradientboosting (right curve) performed almost the same with some marginal win for gradientboosting.

![ROC curve for RFC](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/RFC.png)
![ROC curve for GBC](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/GBC.png)



## Merging Article Titles with the Original CSV File
Here, I have used the url as a key to merge titles to their associated rows in the original CSV file. My goal here, is to use the 'number of shares' column in the [original CSV file](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/OnlineNewsPopularity.csv) and 'titles' column in [the scraped article titles.](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/article_titles_urls.csv)

Before merging them, I had to address a little problem and that was replacing https with http in the newly parsed CSV file. Seems like the original CSV file was from a couple of years ago and that is why the article links were http, and my newly scraped data contained https for article titles.
```python
df_total = pd.read_csv('OnlineNewsPopularity.csv')
df_total = df_total[['url',' shares']]
df_total['url'] = df_total['url'].map(lambda x: str(x)[:-1])
df = pd.read_csv('article_titles_urls.csv', sep='\n', header = None)
titles = df[0].str.split('/,', expand=True)
titles['url']= titles[0].str.replace('https','http')
titles.drop(columns = 0, axis = 1, inplace =True)
titles = titles[['url',1]]

```
After taking care of that, it is time to merge two dataframes on 'url'. I also added a binary column with a condition of 1 for rows that have higher than 1400 shares and 0 for rows with lower than 1400 shares. So, popular articles will be 1s and unpopular articles will be 0s.

```python
titles_numberofshares_df = pd.merge(df_total,titles, how='inner', on=['url'])
titles_numberofshares_df.drop(['url'], inplace=True, axis=1)
final_df=titles_numberofshares_df.drop(titles_numberofshares_df.index[0])
final_df['popular'] = final_df[' shares'].apply(lambda x: 1 if (x >= 1400) else 0)
```
At this point, I will define X matrix which is the titles of articles, and y matrix which is the binary column (popular-unpopular). Splitting to train and test will happen afterward:

```python
from sklearn.model_selection import train_test_split

X = final_df[1].values
y = final_df['popular'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
I will be trying to make a simple LSTM RNN to predict the popularity of an article based on it title. First, some imports:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
Then,some hyperparameters will be chosen based on the data size and some other considerations.

```python
vocab_size = 5000
max_length = 14
embedding_dim = 128
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
```
I will tokenize the train set first and then basically, repeat the same process to tokenize test set. The motivation here, is to convert string to number vectors. The number vectors will be then fed to RNN.


```python
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_train_article_titles_sequences = tokenizer.texts_to_sequences(X_train)
X_train_article_titles_padded = pad_sequences(X_train_article_titles_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

X_test_article_titles_sequences = tokenizer.texts_to_sequences(X_test)
X_test_article_titles_padded = pad_sequences(X_test_article_titles_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
```
Most of the times, it is the best practice to start with very simple RNN. If simple networks don't perform well, additional layers are always easy to be added thanks to Tensorflow Keras:

```python
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
model1.add(tf.keras.layers.LSTM(units = embedding_dim))
model1.add(tf.keras.layers.Dropout(0.2))
model1.add(tf.keras.layers.Dense(1, activation = 'relu'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()
num_epochs = 15
history = model1.fit(X_train_article_titles_padded, y_train, epochs=num_epochs, validation_data=(X_test_article_titles_padded, y_test), verbose=2)
```
You can find the complete code for this part [here.](https://github.com/farzad-yousefi/Assessment-of-online-news-popularity/blob/master/LSTM_RNN_news_popularity.ipynb)






