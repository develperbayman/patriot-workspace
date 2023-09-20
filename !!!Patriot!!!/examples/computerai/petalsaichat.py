import os
import torch
import transformers
import wandb
from flask import Flask, request, render_template
from gtts import gTTS
import speech_recognition as sr
from transformers import BloomTokenizerFast, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BloomTokenizerFast, get_scheduler
from petals import DistributedBloomForCausalLM

app = Flask(__name__)

MODEL_NAME = "bigscience/bloom-7b1-petals"
TUNING_MODE = 'ptune'
NUM_PREFIX_TOKENS = 16
DEVICE = 'cuda'
BATCH_SIZE = 8
LR = 1e-2
WEIGHT_DECAY = 0.0
NUM_SAMPLES = 1000
SEED = 42
MODEL_MAX_LENGTH = 256

tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH
model = DistributedBloomForCausalLM.from_pretrained(
    MODEL_NAME,
    pre_seq_len=NUM_PREFIX_TOKENS,
    tuning_mode=TUNING_MODE
).to(DEVICE)

dataset = load_dataset("bavard/personachat_truecased")

def chunking(examples):
    inputs = [
        "\n-----\n".join(history) + "\n-----\n" + candidate
        for history, candidates in zip(examples["history"], examples["candidates"])
        for candidate in candidates
    ]
    return {"chunks": inputs}

def tokenize(examples):
    outputs = {
        "input_ids": tokenizer(examples["chunks"], padding='max_length', truncation=True)["input_ids"]
    }
    outputs["labels"] = outputs["input_ids"]
    return outputs

tokenized_datasets = (
    dataset
    .map(chunking, batched=True, remove_columns=dataset["train"].column_names)
    .map(tokenize, batched=True, remove_columns=["chunks"])
)

tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
train_dataloader = DataLoader(
    train_dataset.select(list(range(NUM_SAMPLES))),
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=True,
)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
)

wandb.init(
    project="bloom-personachat",
    config={
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_prefix_tokens": NUM_PREFIX_TOKENS,
        "model_name": MODEL_NAME,
        "seed": SEED,
    }
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return response
    return render_template("index.html")

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

def listen_to_activation():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for activation word...")
        audio = r.listen(source)

    try:
        activation_word = r.recognize_google(audio)
        if "computer" in activation_word.lower():
            print("Activation word detected. Please speak your input.")
            listen_for_input()
        else:
            print("Activation word not detected. Listening again...")
            listen_to_activation()
    except sr.UnknownValueError:
        print("Could not understand audio. Listening again...")
        listen_to_activation()

def listen_for_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = r.listen(source)

    try:
        user_input = r.recognize_google(audio)
        print("User input:", user_input)
        response = generate_response(user_input)
        speak_response(response)
    except sr.UnknownValueError:
        print("Could not understand audio. Please repeat.")
        listen_for_input()

def speak_response(response):
    print("Generated response:", response)
    tts = gTTS(text=response, lang='en', tld='ie')
    tts.save("response.mp3")
    os.system("play response.mp3")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

