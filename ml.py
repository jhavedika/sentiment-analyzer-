import streamlit as st
import pickle
import string
import sklearn
import nltk
from  nltk.corpus import stopwords
from sklearn import svm
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
ps=PorterStemmer()
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf=pickle.load(open('vectorizer1.sav','rb'))
model=pickle.load(open('SMV','rb'))
st.title("Sentiment Analysier")
intput_sms=st.text_input("Enter the message")
if st.button('Predict'):
    #1. preprocess
    transformed_sms=transform_text(intput_sms)
    #2. vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3. predict
    result=model.predict(vector_input)[0]
    #4. display
    if result==10:
      st.image('happy.webp',width=3)
      st.header("Positive Sentiment")
    elif result==5:
      st.image('Neutral.webp',width=3)
      st.header("Neutral Sentiment")
    elif result== 0:
      st.image('Sad.webp',width=3)
      st.header("Negative Sentiment")
