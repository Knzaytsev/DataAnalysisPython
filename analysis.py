import nltk
import pymorphy2
import sys
import re
import json
from pandas import DataFrame
from nltk.probability import FreqDist

def getLowTextWithoutPunct(text):
    textWithoutPunct = re.sub(r"[(\ufeff)'\"!,.?-_><«»—a-zA-Z0-9:;–-]", '', text)
    return textWithoutPunct.lower()

def loadStopWords(file):
    with open(file, mode='r') as stop_words:
        return json.load(stop_words)

def getTokenizedTextWithoutStopWords(text, language):
    tokens = nltk.word_tokenize(text, language)
    stop_words = loadStopWords("stopwords.json")
    return [w for w in tokens if not w in stop_words]

def getNormalizedWords(words):
    normalized_words = []
    morph = pymorphy2.MorphAnalyzer()
    for w in words:
        p = morph.parse(w)[0].normal_form
        normalized_words.append(p)
    return normalized_words

def getSortedWordsWithFrequency(words):
    fdist = FreqDist(words)
    return sorted(fdist.items(), key = lambda x: x[1], reverse=True)

def exportResults(file, wordsWithFrequency):
    words = []
    count = []
    for w in wordsWithFrequency:
        words.append(w[0])
        count.append(w[1])
    Data = {'Слово': words,
            'Количество': count}
    df = DataFrame(Data, columns=['Слово', 'Количество'])
    df.to_excel(file, index=None, header=True)

nltk.download('punkt')
nltk.download('stopwords')
if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8", mode="r") as textfile:
        text = textfile.read()
        clear_text = getLowTextWithoutPunct(text)
        tokens = getTokenizedTextWithoutStopWords(clear_text, 'russian')
        normalized_words = getNormalizedWords(tokens)
        sorted_counts = getSortedWordsWithFrequency(normalized_words)
        exportResults(r'output.xlsx', sorted_counts)
