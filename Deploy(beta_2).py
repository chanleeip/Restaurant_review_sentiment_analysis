import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=1)

# Download NLTK data
nltk.download('stopwords')

# Load the saved SVM model from a file
with open('restaurant_review_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Create a function to preprocess the input string
def preprocess_input(input_str):
    # Remove non-alphabetic characters
    input_str = re.sub('[^a-zA-Z]', ' ', input_str)

    # Convert to lowercase
    input_str = input_str.lower()

    # Split into words
    words = input_str.split()

    # Remove stopwords and stem the remaining words
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('no')
    words = [ps.stem(word) for word in words if word not in stop_words]

    # Join the words back into a string
    preprocessed_input = ' '.join(words)

    return preprocessed_input


# Get input from the user
input_str = "not good"

# Preprocess the input
preprocessed_input = [preprocess_input(input_str)]
x=tf.fit_transform(preprocessed_input).toarray()
x.reshape((1, -1))
# Make a prediction using the loaded SVM model
prediction = model.predict(x)

# Print the prediction

