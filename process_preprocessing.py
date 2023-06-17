
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import re



def cleaningText(text):
    # text= text.split('_', 1)[0]
    # remove mentions
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # remove hashtag
    text = re.sub(r'RT[\s]', '', text)  # remove RT
    text = re.sub(r"http\S+", '', text)  # remove link
    text = re.sub(r'[0-9]+', '', text)  # remove numbers

    text = text.replace('\n', ' ')  # replace new line into space
    text = text.replace(r'[^\x00-\x7F]+', ' ')
    text = text.replace(r'^\s+|\s+?$', '')
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    # remove all punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove characters space from both left and right text
    text = text.strip(' ')
    #remove non-ascii characters
    ascii_chars = set(string.printable)
    return ''.join(
        filter(lambda x: x in ascii_chars, text)
    )


def casefoldingText(text):  # Converting all the characters in a text into lower case
    text = text.lower()
    return text


def tokenizingText(text):  # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text


def filteringText(text):  # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text


def stemmingText(text):  # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text


def toSentence(list_words):  # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence


# Location
def casefoldingLocation(location):
    location = location.lower()
    return location


def removeStringAfterSymbol(location):
  location = location.split(',', 1)[0]
  location = location.split('-', 1)[0]
  return location


def removeUnusedString(location):
  location = location.replace("dki", "")
  location = location.replace("jawa", "")
  location = location.replace("timur", "")
  location = location.replace("selatan", "")
  location = location.replace("barat", "")
  location = location.replace("utara", "")
  location = location.replace("tengah", "")
  location = location.replace("capital region", "")
  location = location.replace("kota", "")
  location = location.replace("pusat", "")
  location = location.replace("indonesia", "")
  return location


def prepocessingText(text):
    text['sentiments'] = text.apply(cleaningText)
    text['sentiments'] = text['sentiments'].apply(casefoldingText)
    text['sentiments'] = text['sentiments'].apply(tokenizingText)
    text['sentiments'] = text['sentiments'].apply(filteringText)
    # text['sentiments'] = text['sentiments'].apply(stemmingText)
    text['sentiments'] = text['sentiments'].apply(toSentence)
    return text['sentiments']

def preprocessingLocation(location) :
    location['locations'] = location.apply(casefoldingLocation)
    location['locations'] = location['locations'].apply(removeStringAfterSymbol)
    location['locations'] = location['locations'].apply(removeUnusedString)
    return location['locations']
