import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import flask_
import numpy
with open('vectorizer00.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
# Open the pkl file for reading
with open('restaurant_review_sentiment_model_o.pkl', 'rb') as f:
    # Load the contents of the file into a variable
    data = pickle.load(f)
corpus = []
ps = PorterStemmer()
stop_words = stopwords.words('english')
stop_words.remove('no')

# Now you can use the data variable in your program
sttring=input()
sttring=re.sub('[^a-zA-Z]',' ',sttring)
sttring=sttring.lower()
sttring=sttring.split()
sttring=[ps.stem(j) for j in sttring if not j in set(stop_words)]
sttring=' '.join(sttring)
corpus.append(sttring)

# print(corpus)
xoxo=vectorizer.transform(corpus)
predict=data.predict(xoxo.toarray())
if predict==1:
    print("positive")
else:
    print("negative")
