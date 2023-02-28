import pickle
import nltk
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model from a file
with open('restaurant_review_sentiment_model_o.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess the input string
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.remove('no')

input_str = 'good '
input_str = re.sub('[^a-zA-Z]', ' ', input_str)
input_str = input_str.lower()
input_str = input_str.split()
input_str = [ps.stem(word) for word in input_str if word not in stop_words]
input_str = ' '.join(input_str)

# Convert the input string to a feature vector
input_vector = np.array([[len(input_str), input_str.count('!')]])

# Make the prediction using the loaded model
prediction = model.predict(input_vector,)

# # Print the prediction
# if prediction[0] == 1:
#     print('The restaurant review is positive.')
# else:
#     print('The restaurant review is negative.')
