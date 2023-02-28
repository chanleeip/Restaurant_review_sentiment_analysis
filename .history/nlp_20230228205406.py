import pandas as pd
import nltk
import re
data_sheet=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[ ]
ps=PorterStemmer()
stop_words=stopwords.words('english') 
stop_words.remove('no')
for i in range (0,1000,1):
    sttring=re.sub('[^a-zA-Z]',' ',data_sheet['Review'][i])
    sttring=sttring.lower()
    sttring=sttring.split()
    sttring=[ps.stem(j) for j in sttring if not j in set(stop_words)]
    sttring=' '.join(sttring)
    corpus.append(sttring)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=2000)
x=tf.fit_transform(corpus).toarray()
y=data_sheet.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(  x, y, test_size=0.3)
from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test,Y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
import pickle

# Save the model to disk
filename = 'restaurant_review_sentiment_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)

