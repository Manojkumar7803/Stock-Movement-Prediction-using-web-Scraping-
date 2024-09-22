# Stock-Movement-Prediction-using-web-Scraping-
//machine learning project that predicts stock movement based on data scraped from social media platforms like Twitter.
pip install -r requirements.txt
python scrape_twitter.py
python train_model.py
//scrap data fronsocial media twitter
import tweepy

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Scrape tweets using hashtags
tweets = tweepy.Cursor(api.search_tweets, q='#stocks', lang="en").items(100)
for tweet in tweets:
    print(tweet.text)
//Data Processing 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores("Tesla stock is going up!")
print(sentiment)
//train the machine learning model 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample data - replace with your own features
X = [[0.1, 0.2, 0.5], [0.3, 0.6, 0.9], ...] # Feature matrix (sentiment scores, mentions frequency)
y = [1, 0, ...] # Target labels (1 for price increase, 0 for price drop)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
//Model Evaluation 
from sklearn.metrics import accuracy_score, precision_score, recall_score

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
