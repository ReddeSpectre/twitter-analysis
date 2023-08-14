import numpy as np 
import random
import pandas as pd 
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
import openai
from sklearn.model_selection import train_test_split
import concurrent.futures
import streamlit as st
import nltk

keyinput = 0
while (keyinput == 0):
    #Get API Key from user
    st.write("In order to continue the demonstration, please input a valid OpenAI API Key. An Error Message will appear later in the code until a valid key is applied:")
    key = st.text_input("API Key", "[Insert API Key Here]")
    keyinput = 1
st.write("The remaining code may take a few minutes to load.")
openai.api_key = key

st.title("ML Model Output Validation using Generative AI")
st.write("If we want to use Generative AI to validate the output of a ML model, the Generative AI must operate at or above the precision of the ML Model. In order to evaluate if this is possible, we will be comparing a Basic ML Model to a Basic Generative AI Model.") 
st.write("The experiment begins with training and testing a sentiment analysis model. We'll be using a SVC model for this case study.")

# Setup dataset 2 for use
review = pd.read_csv(r'ChatGPT related Twitter Dataset.csv')

positive = review[review['labels'] == 'good'][:25000]
negative = review[review['labels'] == 'bad'][:10000]
neutral = review[review['labels'] == 'neutral'][:45000]

review_bal = pd.concat([positive, negative, neutral])

# Set up the GPT Dataset for direct comparison
gpt_dataset = review_bal.sample(100, random_state=42)
gpt_dataset = gpt_dataset.dropna()

# Set up the Sentiment Dataset
sentiment_dataset = review_bal[~review_bal.index.isin(gpt_dataset.index)]
sentiment_dataset = sentiment_dataset[~sentiment_dataset.index.isin(a_b_dataset.index)]
sentiment_dataset = sentiment_dataset.dropna()

st.write("The two datasets we will be using in this case study(for the inital training and for the later comparison test) are as follows:")

st.write("Initial Training:")
st.write(sentiment_dataset)

st.write("Later Comparison:")
st.write(gpt_dataset)

# Remove punctuation from datasets
sentiment_dataset['tweets'] = sentiment_dataset['tweets'].str.replace('[{}]'.format(string.punctuation), '')
gpt_dataset['tweets'] = gpt_dataset['tweets'].str.replace('[{}]'.format(string.punctuation), '')
a_b_dataset['tweets'] = a_b_dataset['tweets'].str.replace('[{}]'.format(string.punctuation), '')

keyinput = 0
while (keyinput == 0):
    #Get API Key from user
    st.write("In order to continue the demonstration, please input a valid OpenAI API Key. An Error Message will appear later in the code until a valid key is applied:")
    key = st.text_input("API Key", "[Insert API Key Here]")
    keyinput = 1
st.write("The remaining code may take a few minutes to load.")
openai.api_key = key

# Function to generate GPT-3 response
def generate_gpt3_response(tweets):
    prompt = f"Tweet: {tweets}\n"
    response = openai.Completion.create(
        model='curie:ft-personal:curie-sentiment-analysis-2023-07-06-20-16-11',
        #model='davinci:ft-personal:davinci-sentiment-analysis-2023-07-06-23-55-02',
        #ft-3gc5aDo7de0oSbBtrTdR4Tbf
        prompt="Analyzing sentiment: This text is [positive/negative/neutral].",
        max_tokens=100
    )
    generated_text = response.choices[0].text.strip()
    return generated_text

#Setup the Train/Test Split for the SA Model
from sklearn.model_selection import train_test_split

train,test = train_test_split(sentiment_dataset,test_size =0.5,random_state=42)

train_x, train_y = train['tweets'], train['labels']
test_x, test_y = test['tweets'], test['labels']

# Setup the Train/Test Split for the GPT SA Model
gpt_train,gpt_test = train_test_split(gpt_dataset, test_size =0.5,random_state=42)
gpt_test_x,gpt_test_y = gpt_test['tweets'], gpt_test['labels']

#Setup the A/B Testing Split
a_test,b_test = train_test_split(a_b_dataset, test_size =0.5,random_state=42)
a_test_x,a_test_y = a_test['tweets'], a_test['labels']
b_test_x,b_test_y = b_test['tweets'], b_test['labels']

#Implement Stop Words, vectorize data

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)
gpt_test_x_vector = tfidf.transform(gpt_test_x)
a_test_x_vector = tfidf.transform(a_test_x)
b_test_x_vector = tfidf.transform(b_test_x)

#Fit the Data to an SVC Model

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

#Transform Train_x data for later testing. 

pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                  index=train_x.index,
                                  columns=tfidf.get_feature_names_out())

# Store the training and testing accuracy during each epoch
train_accuracy = []
test_accuracy = []

# Evaluate the model on training and testing data
train_predictions = svc.predict(train_x_vector)
test_predictions = svc.predict(test_x_vector)

# Calculate accuracy scores
train_acc = accuracy_score(train_y, train_predictions)
test_acc = accuracy_score(test_y, test_predictions)

train_accuracy.append(train_acc)
test_accuracy.append(test_acc)

st.write("After completing the Training of the Model, the results for training and testing accuracy are as follows:")
# Print the training and testing accuracy
st.write("Training Accuracy:")
st.caption(train_acc)

st.write("Testing Accuracy:")
st.caption(test_acc)

st.bar_chart(data = [train_accuracy, test_accuracy], x = None, y = None, width = 0, height = 0, use_container_width = True)
st.caption(svc.score(test_x_vector, test_y))

st.write("SVC Score:")
st.caption(svc.score(test_x_vector, test_y))

from sklearn.metrics import f1_score

f1_score = f1_score(test_y,svc.predict(test_x_vector),
          labels = ['good', 'neutral', 'bad'],average=None)

st.write("F1 Score:")
st.caption(f1_score)

from sklearn.metrics import classification_report

st.write("Classification Report:")
st.caption(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels = ['good', 'neutral', 'bad']))

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y,
                           svc.predict(test_x_vector),
                           labels = ['good', 'neutral', 'bad'])

st.write("Confusion Matrix:")
st.caption(conf_mat)

st.write("For practical use, this model does not meet acceptable standards, but for the sake of the case, basic training is acceptable. The same standard of training was done for the GPT-3 Model we will use in the direct comparison: a Custom trained Curie GPT-3 Model.")

st.write("The next step of our case study requires us to, with the help of the Vader Sentiment Lexicon, run the sentiment analysis through our custom GPT-3 Model and compare the results to our previously trained SVC model")

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Generate GPT-3 responses for the testing data
gpt3_responses = []
error_counter = 0

for _ in range(4):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_review = {executor.submit(generate_gpt3_response, review): review for review in gpt_test_x_vector}
        for future in concurrent.futures.as_completed(future_to_review):
            review = future_to_review[future]
            try:
                response = future.result()
                gpt3_responses.append(response)
            except Exception as e:
                print(f"Error processing review: {review}. Exception: {e}")
                response = 'neutral'
                gpt3_responses.append(response)
                error_counter += 1


# Evaluate GPT-3 responses
gpt3_predictions = []
for response in gpt3_responses:
    # Perform sentiment analysis using VADER
    sentiment = sia.polarity_scores(response)
    if sentiment['compound'] >= 0.05:
        gpt3_predictions.append('good')
    elif sentiment['compound'] <= -0.05:
        gpt3_predictions.append('bad')
    else:
        gpt3_predictions.append('neutral')

# Ensure the sizes of gpt_test_predictions and test_y match
for _ in range(4):
    num_samples = min(len(gpt_test_predictions), len(test_y))
    gpt_test_predictions = gpt_test_predictions[:num_samples]
    test_y = test_y[:num_samples]

# Compare results between Sentiment Analysis Model and GPT-3
comparison_results = []
for i in range(len(test_y)):
    sa_prediction = gpt_test_predictions[i]
    gpt3_prediction = gpt3_predictions[i]
    actual_label = test_y.iloc[i]
    comparison_results.append((gpt3_prediction, sa_prediction, actual_label))

# Convert comparison_results to a DataFrame
df = pd.DataFrame(comparison_results, columns=['gpt3_prediction', 'sa_prediction', 'actual_label'])

# Calculate accuracy scores
gpt3_accuracy = (df['gpt3_prediction'] == df['actual_label']).mean()
sa_accuracy = (df['sa_prediction'] == df['actual_label']).mean()

st.bar_chart(data = [gpt3_accuracy, sa_accuracy], x = None, y = None, width = 0, height = 0, use_container_width = True)
st.caption(svc.score(test_x_vector, test_y))

st.write("Our final totals for accuracy are as follows:") 

st.write('GPT-3 Accuracy: ')
st.caption(gpt3_accuracy)

st.write('SA Accuracy: ')
st.caption(sa_accuracy)

st.write("This last line is placeholder text for the conclusion.")

