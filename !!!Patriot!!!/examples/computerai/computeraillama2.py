import os
import threading
import time
from gtts import gTTS
import speech_recognition as sr
from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Hugging Face model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

doListenToCommand = True
listening = False
despedida = ["Goodbye", "goodbye", "bye", "Bye", "See you later", "see you later"]

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def submit():
    global doListenToCommand

    usuario = request.form["user_input"]
    if usuario in despedida:
        return "Goodbye!"
    else:
        response = generate_response(usuario)

        texto = str(response)
        tts = gTTS(texto, lang='en', tld='ie')
        tts.save("audio.mp3")

        doListenToCommand = False
        time.sleep(1)
        os.system("play audio.mp3")
        doListenToCommand = True

        return response

@app.route("/", methods=["POST"])
def index():
    user_input = request.form["user_input"]
    response = generate_response(user_input)
    return response

def listen_to_command():
    global doListenToCommand
    global listening

    if doListenToCommand == True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            listening = True
            audio = r.listen(source)
            listening = False

        try:
            command = r.recognize_google(audio)
            print("You said:", command)

            passed_commands = {
                "submit": submit
            }
            process_commands(passed_commands, command)

        except sr.UnknownValueError:
            print("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service:", str(e))

        listen_to_command()
        listening = False

def process_commands(commands, user_input):
    # Define your custom command processing logic here
    pass

if __name__ == "__main__":
    start_listening_thread = threading.Thread(target=listen_to_command)
    start_listening_thread.daemon = True
    start_listening_thread.start()

    app.run(host="0.0.0.0", port=5024, debug=True)
