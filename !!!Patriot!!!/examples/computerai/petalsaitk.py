import os
import tkinter as tk
from gtts import gTTS
import speech_recognition as sr
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

# Initialize the Tkinter window
root = tk.Tk()
root.title("Petals Chatbot")

# Create a Text widget for displaying messages
text_widget = tk.Text(root)
text_widget.pack()

# Initialize the chatbot components
MODEL_NAME = "bigscience/bloom-7b1-petals"
DEVICE = 'cuda'
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Function to generate a response from user input
def generate_response(user_input):
    inputs = tokenizer([f"{user_input}\n-----\n"], return_tensors='pt')['input_ids'].to(DEVICE)
    with model.inference_session(max_length=512) as sess:
        outputs = model.generate(
            inputs,
            temperature=0.6,
            do_sample=True,
            top_k=100,
            max_new_tokens=50,
            session=sess,
        )
        bloom_answer_token = tokenizer.decode(outputs[0, -1:])
        return bloom_answer_token

# Function to speak the response
def speak_response(response):
    tts = gTTS(text=response, lang='en')
    tts.save("response.mp3")
    os.system("play response.mp3")

# Function to handle sending user message
def send_user_message():
    user_input = user_input_entry.get()
    add_message("You: " + user_input, True)
    response = generate_response(user_input)
    add_message("Bot: " + response, False)
    speak_response(response)

# Function to add a message to the display
def add_message(message, is_user):
    tag = "user" if is_user else "bot"
    text_widget.insert(tk.END, "\n" + message, tag)
    text_widget.see(tk.END)

# Entry widget for user input
user_input_entry = tk.Entry(root)
user_input_entry.pack()

# Button to send user message
send_button = tk.Button(root, text="Send", command=send_user_message)
send_button.pack()

# Styling for user and bot messages
text_widget.tag_configure("user", foreground="blue")
text_widget.tag_configure("bot", foreground="green")

root.mainloop()
