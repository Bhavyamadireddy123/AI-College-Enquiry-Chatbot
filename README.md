# AI-Powered College Enquiry Chatbot

## Introduction

The AI-Powered College Enquiry Chatbot is a deep learning based application developed to assist students in accessing academic and timetable-related information through a conversational interface. The system is capable of understanding user queries and generating appropriate responses using Natural Language Processing (NLP) and TensorFlow.

The chatbot provides an interactive platform where students can ask questions related to class schedules, laboratory sessions, subjects, and academic activities. The web application was developed using Streamlit with a modern and user-friendly interface.

---

## Objectives

- To design an intelligent chatbot for handling college-related queries.
- To automate timetable and subject enquiry processes.
- To implement Natural Language Processing techniques for understanding user input.
- To develop a deep learning model for accurate response prediction.
- To create an interactive web application for real-time communication.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Natural Language Processing (NLP)
- NumPy
- Pandas
- Scikit-learn
- NLTK

---

## Methodology

The chatbot system was developed in multiple stages including data preprocessing, text tokenization, model training, and web application development.

Initially, the dataset containing question-answer pairs was collected and preprocessed. NLP techniques such as stopword removal, lemmatization, lowercasing, and tokenization were applied to improve model performance.

The processed text data was then converted into numerical sequences using a tokenizer. A deep learning model based on Sequential architecture was trained using TensorFlow and Keras. The trained model predicts responses based on user queries entered through the Streamlit interface.

---

## Deep Learning Model

The chatbot model consists of the following layers:

- Embedding Layer
- Conv1D Layer
- Global Max Pooling Layer
- Dense Layer
- Dropout Layer
- Softmax Output Layer

The model was trained using categorical classification techniques to identify the most suitable response for a given query.

---

## Features

- Interactive chatbot interface
- Real-time response generation
- NLP-based query understanding
- Deep learning response prediction
- Modern web interface using Streamlit
- High prediction accuracy
- User-friendly design

---

## Project Structure

AI-College-Chatbot/

├── dataset/

├── notebook/

├── app.py

├── chatbot_model.keras

├── tokenizer.pkl

├── label_encoder.pkl

├── requirements.txt

├── README.md

└── train_chatbot.ipynb

---

## Model Performance

The chatbot was trained using an augmented dataset to improve prediction accuracy and response quality. Data augmentation techniques were used to generate multiple variations of user questions, enabling the model to learn diverse query patterns effectively.

The trained model achieved high training and validation accuracy during testing.

---

## Web Application

The web application was developed using Streamlit. The interface includes a chatbot-style interaction system where users can enter queries and receive responses instantly.

The application features:

- Gradient-based modern UI
- Chat-style response display
- Interactive buttons
- Responsive design
- Session-based chat history

---

## Future Enhancements

The project can be further improved by adding:

- Voice-based interaction
- Database integration
- Multi-language support
- Faculty information module
- Attendance enquiry system
- Cloud deployment support

---

## Conclusion

The AI-Powered College Enquiry Chatbot successfully demonstrates the application of deep learning and NLP techniques in developing an intelligent academic assistance system. The project provides an efficient and interactive method for students to access timetable and subject-related information through a conversational web application.

---
