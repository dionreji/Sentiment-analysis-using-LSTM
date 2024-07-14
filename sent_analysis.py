'''
Importing essential libraries and functions
'''
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import io
import json
import matplotlib.pyplot as plt

df = pd.read_csv('IMDB_Dataset.csv')
print(df.head())
print(df.shape)

missing_data = df.isnull().sum()
print("Missing data in each column:\n", missing_data)


def preprocessing(sentence):
    # First make the sentence lowercase
    sentence = sentence.lower()
    
    # Remove all html tags from the sentence i.e replace anything between <> with space
    # Hint use Regular Expression i.e. re.sub()
    pattern = r"<.*?>"
    replacement = " "
    sentence = re.sub(pattern, replacement, sentence)
    
    # Remove all special characters i.e. anything other than alphabets and numbers. Replace them with space
    pattern = r"\W+"
    sentence = re.sub(pattern, replacement, sentence)
    # Remove all single characters i.e. a-z and A-Z and Replace them with space
    pattern = r"\b[a-zA-Z]\b"
    sentence = re.sub(pattern, replacement, sentence)
    
    # Remove all multiple spaces and replace them with single space
    pattern = r"\s+"
    sentence = re.sub(pattern, replacement, sentence)
    
    # Use the nltk library to remove all stopwords from the sentence
    # stopwords are the words like and, the, is, are etc.
    nltk.download('punkt')
    words = nltk.word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [word for word in words if word.lower() not in stop_words]

    sentence = ' '.join(filtered_sentence)
    
    # return the sentence
    return sentence


# # Test case
# sample_sentence = "This is <b>an example</b> &&   sentence with <i>html tags</i> and special characters like !@$#"
# processed_sentence = preprocessing(sample_sentence)

# # Print results
# print("Original Sentence:")
# print(sample_sentence)
# print("\nProcessed Sentence:")
# print(processed_sentence)

X = []
# Call the preprocessing function for each review in the dataframe and
# save the results in a new list of preprocessed_reviews
for review in df['review']:
    X.append(preprocessing(review))
# This list will be your input to the neural network
# We will call this list as X from now on

# Convert sentiment column in the dataframe to numbers
# Convert positive to 1 and negative to 0 and store it in numpy array
# We will call this numpy array as y from now on
y = np.array([1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']])


#Split the data into training and testing (80-20 ratio)
# The train set will be used to train our deep learning models 
# while test set will be used to evaluate how well our model performs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the tokenizer
word_tokenizer = Tokenizer()

# Fit the tokenizer on the training data (X_train)
# X_train = np.array(X_train)
word_tokenizer.fit_on_texts(X_train)

#Convert training data to sequences of integers
# Hint: Use texts_to_sequences method
X_train = word_tokenizer.texts_to_sequences(X_train)

# Convert test data to sequences of integers
# Hint: Use texts_to_sequences method
X_test = word_tokenizer.texts_to_sequences(X_test)

# Saving the tokenizer in a json file 
tokenizer_json = word_tokenizer.to_json()
with io.open('b3_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    
# Vocab_length is the number of unique words in our dataset
# Adding 1 to store dimensions for words for which no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1

# Padding all reviews to be of same length 'maxlen' words
maxlen = 100
# Try different dimensions like 50, 100, 200 and 300


# TODO: Pad the training data sequences
# Hint: Use pad_sequences with 'post' padding and maxlen=maxlen
padded_X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

# TODO: Pad the test data sequences
# Hint: Use pad_sequences with 'post' padding and maxlen=maxlen
padded_X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Initialize an empty dictionary for embeddings
embeddings_dictionary = dict()

# Open the GloVe file (a2_glove.6B.100d.txt) with utf-8 encoding
glove_file = open('glove_embeddings.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# TODO : Create an embedding matrix where each row corresponds to the index of the
# unique word in the dataset and each column corresponds to the word vector
# in the GloVe embedding 
# So the matrix will have vocab_length rows and maxlen columns

embeddings_matrix = np.zeros((vocab_length, maxlen))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embeddings_matrix[index] = embedding_vector



# Training the model with LSTMs
from keras.layers import LSTM
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_length, maxlen, input_length=maxlen, weights=[embeddings_matrix], trainable=False))
model.add(LSTM(128))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# TODO 3: Train the model with the training data. 
# Try using different batch sizes and number of epochs to see how the model performs.
history = model.fit(padded_X_train, y_train, batch_size=64, epochs=5, validation_data=(padded_X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(padded_X_test, y_test)

# Print the accuracy of the model
print(f"Accuracy of the model is: {accuracy:.4f}")

# Plot training history (optional)
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('loss.png')

# Making Live predictions on IMDB reviews using LSTM

test = "I absolutely loved this movie! The acting was great and the story was very compelling. Definitely recommend it!"
    
# Preprocess the data as you did in the previous assignment.
test = preprocessing(test)

# from keras_preprocessing.text import tokenizer_from_json

# Tokenize the data using the tokenizer_from_json() function. You may use another tokenizer if you wish.
test = word_tokenizer.texts_to_sequences([test])
# Pad the data using the pad_sequences() function.
test = pad_sequences(test, padding='post', maxlen=maxlen)

# Make predictions on the data using the model.predict() function.
prediction = model.predict(test)

#  For each review, if the prediction is stored in variable, 'prediction', then np.round(prediction*10,1) will give you the predicted rating
# Store this back to csv and compare the predicted ratings with the actual ratings.
rating = np.round(prediction*10, 1)
print(rating)