import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os

def chat(msg):
    with open('dataset_process.json', 'r') as f:
        intents = json.load(f)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    FILE = "data.pth"
    data = torch.load(FILE, map_location=map_location)
    
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]
    
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()
    
    # Tokenize the received message
    msg = tokenize(msg)
    X = bag_of_words(msg, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Predict the appropriate tags for the input message
    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]

    # Find the probability of each tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If the probablity of the predicted tag is greater than 75% then reply with respective message
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # Create reply
                return (random.choice(intent['responses']).format(msg))
    
    # If the probablity of the predicted tag is less than 75% then reply with error message and send a mail to manager
    else:
        return ("I do not understand. One of our executives would contact you soon.".format(msg))

while True:
    msg = input("Enter Your Message: ")
    if msg == "exit":
        break
    else:    
        print(chat(msg))
