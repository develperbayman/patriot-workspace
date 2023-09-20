from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import os
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large", return_dict=True)

chat_history = []

def generate_bot_response(user_input, chat_history):
    chat_history.append({"user": user_input, "bot": ""})

    input_ids = tokenizer.encode(" ".join([entry["user"] for entry in chat_history]), return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    chat_history[-1]["bot"] = bot_response

    return bot_response

@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = generate_bot_response(user_input, chat_history)

        # Convert bot response to audio
        tts = gTTS(text=bot_response, lang="en")
        tts.save("response.mp3")

        # Play the audio response
        os.system("mpg321 response.mp3")
        os.remove("response.mp3")

        return render_template("chat.html", chat_history=chat_history)
    else:
        return render_template("chat.html", chat_history=chat_history)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5021)
