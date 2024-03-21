from flask import Flask, request, jsonify
import speech_recognition as sr
from nlp import process_input

app = Flask(__name__)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

@app.route('/chat', methods=['POST'])
def chat():
    # Receive user input from the request
    if request.method == 'POST':
        if 'audio' in request.files:
            # If audio file is sent, use speech recognition to convert it to text
            audio_file = request.files['audio']
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return jsonify({'response': "Sorry, I couldn't understand your audio."})
        else:
            # If text is sent, directly use it
            user_input = request.form['text']
        
        # Process user input using NLP module
        response = process_input(user_input)
        
        # Return the response
        return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
