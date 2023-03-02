from flask import *
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy
app = Flask(__name__)



@app.route('/')
def home():
    title='Welcome'
    return render_template('home.html')

@app.route('/process_form',methods=['GET','POST'])
def process_form():
    msg =" "
    if request.method == 'POST':
        nithin = request.form['name']
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
        sttring = nithin
        sttring = re.sub('[^a-zA-Z]', ' ', sttring)
        sttring = sttring.lower()
        sttring = sttring.split()
        sttring = [ps.stem(j) for j in sttring if not j in set(stop_words)]
        sttring = ' '.join(sttring)
        corpus.append(sttring)

        # print(corpus)
        xoxo = vectorizer.transform(corpus)
        predict = data.predict(xoxo.toarray())
        if predict == 1:
            msg ="positive"
        else:
            msg ="negative"
    return render_template('result.html',opinion=msg)

if __name__ == '__main__':
    app.run(debug=True)
