import snscrape.modules.twitter as sntwitter
import pandas as pd
import warnings

def scrapping(keyword, total):
	warnings.filterwarnings('ignore')
	# Define the search query
	query = keyword

	# Set the number of tweets you want to scrape
	limit = total
	tweets_get = []
	# Crawl the tweets
	tweets = sntwitter.TwitterSearchScraper(query).get_items()

	# Process the scraped tweets
	for tweet in tweets:
			# Extract relevant information from the tweet\
			if len(tweets_get) == limit:
				break
			else:
				tweets_get.append([tweet.date, tweet.user.username, tweet.content])
	df = pd.DataFrame(tweets_get, columns = ['datetime', 'username', 'content'])
	return df
	