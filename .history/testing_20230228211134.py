import pickle

# Load the trained model
with open('restaurant_review_sentiment_model  ', 'rb') as file:
    model = pickle.load(file)

# Define a function to preprocess input data
def preprocess(sentence):
    # Implement your preprocessing steps here
    return processed_sentence

# Get input sentence from user
input_sentence = input("Enter a restaurant review: ")

# Preprocess the input sentence
preprocessed_input = preprocess(input_sentence)

# Apply the model to the preprocessed input data
prediction = model.predict(preprocessed_input)

# Output the results to the user
if prediction == 1:
    print("Positive review")
else:
    print("Negative review")
