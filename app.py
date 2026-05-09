import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import re
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK Data
nltk.download('stopwords')
nltk.download('wordnet')
#Load Model Files
model = tf.keras.models.load_model("chatbot_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# NLP Setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
max_len = 20
# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)
# Prediction Function
def get_model_response(question):
    question = clean_text(question)
    seq = tokenizer.texts_to_sequences([question])
    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post"
    )
    preds = model.predict(padded)
    label_idx = np.argmax(preds, axis=1)
    response = label_encoder.inverse_transform(label_idx)[0]
    return response
# Streamlit Page Config
st.set_page_config(
    page_title="AI College Chatbot",
    page_icon="🤖",
    layout="wide"
)
# Custom CSS Styling
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 25%,
        #6dd5ed 50%,
        #2193b0 75%,
        #cc2b5e 100%
    );
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}
/* Background Animation */
@keyframes gradientShift {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
/* Title Box */
.title-box {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(31,38,135,0.37);
}
.title-box h1 {
    font-size: 50px;
    font-weight: bold;
}
.title-box p {
    font-size: 20px;
}
/* Chat Container */
.chat-container {
    background: rgba(255,255,255,0.20);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 30px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 8px 32px rgba(31,38,135,0.37);
}
/* Chat Bubbles */
.chat-bubble {
    padding: 15px 20px;
    border-radius: 20px;
    margin: 15px 0;
    max-width: 75%;
    animation: slideIn 0.4s ease;
}
.user-bubble {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );
    color: white;
    margin-left: auto;
    text-align: right;
    border-bottom-right-radius: 5px;
}
.bot-bubble {
    background: rgba(255,255,255,0.95);
    color: #333;
    margin-right: auto;
    border-bottom-left-radius: 5px;
}
/* Animation */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
/* Input Box */
.stTextInput > div > div > input {
    border-radius: 15px !important;
    padding: 12px !important;
    background: rgba(255,255,255,0.9) !important;
    border: none !important;
}
/* Button */
.stButton > button {
    background: linear-gradient(
        135deg,
        #ff512f,
        #dd2476
    );
    color: white;
    border-radius: 15px;
    padding: 10px 25px;
    border: none;
    font-weight: bold;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        135deg,
        #24c6dc,
        #514a9d
    );
}
/* Footer */
.footer {
    text-align: center;
    color: white;
    margin-top: 50px;
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
# Header
st.markdown("""
<div class="title-box">
    <h1>🤖 AI College Enquiry Chatbot</h1>
    <p>Deep Learning Based Student Assistant</p>
</div>
""", unsafe_allow_html=True)
# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
        <div style='text-align:center; color:white; padding:40px;'>
        <h3>👋 Welcome!</h3>
        <p>Ask your college timetable or subject-related questions.</p>
        </div>
        """, unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"<div class='chat-bubble user-bubble'><b>👤 You:</b> {message}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-bubble bot-bubble'><b>🤖 Chatbot:</b> {message}</div>",
                unsafe_allow_html=True
            )
    user_input = st.text_input(
        "Ask your question",
        placeholder="Example: What subject is scheduled on Monday?"
    )
    colA, colB = st.columns([3,1])
    with colA:
        send_button = st.button("🚀 Ask Chatbot")
    with colB:
        clear_button = st.button("🧹 Clear")
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    if send_button and user_input.strip() != "":
        bot_reply = get_model_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
# Footer
st.markdown("""
<div class="footer">
🎓 Developed using TensorFlow, NLP and Streamlit
</div>
""", unsafe_allow_html=True)
