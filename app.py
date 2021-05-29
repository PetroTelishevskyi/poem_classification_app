import streamlit as st
import pickle
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


model = load_model('models\\model_lstm.h5')
with open('models\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def predict(text):
    MAX_SEQUENCE_LENGTH = 100
    text = remove_punctuation(text)
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    return pred


def main():
    st.title("Poem Classifer by theme ML App")
    
    activities = ['Prediction', 'Info']
    choise = st.sidebar.selectbox("Choose Activity", activities)
 
    if choise == 'Prediction':
        st.subheader("ML and NLP with Streamlit")
        st.info('Prediction poem`s theme with nerual network')

        st.write("======================================================")
        poem_text = st.text_area('Enter Poem','Type here')
        
        if st.button('Classify'):
            pred = predict(poem_text)
            st.write("======================================================")
            st.write("Prediction for all classes:")
            st.write('Death:', round(pred[0][0]*100),"%", 'Love:', round(pred[0][1]*100),"%", 'Nature:', round(pred[0][2]*100),"%")
            poem_id= pred.argmax(axis=1)[0]
            if poem_id == 0:
                poem = 'Death'
            elif poem_id ==1:
                poem = 'Love'
            elif poem_id ==2:
                poem = 'Nature'
            st.write("======================================================")
            st.write("Prediction for result:")
            st.write("Suppose", round(max(pred[0])*100),"%", poem)
            st.write("======================================================")
            
           
    if choise == 'Info':
        st.subheader("Streamlit App")
        st.info('Information about this applipoemion')
        st.write("Our web applipoemion used a neural network of the LSTM type. The following is an image of the model used.")
        st.image("images\\image.jpg") 
        st.write("After training the models got so accurate. The training was carried out for 10 epochs.")
        st.image("images\\image1.jpg")
        st.write("Test options for training the model gave 63% accuracy. This is shown in the image.") 
        st.image("images\\image2.jpg") 


if __name__=='__main__':
    main()