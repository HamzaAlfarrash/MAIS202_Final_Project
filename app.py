import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template, request

model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# chatting 5 times with Top K sampling & tweaking temperature
def predictor(text):
    # take user input
    # text = input(">> You:")
    # encode the input and add end of string token
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    # concatenate new user input with chat history (if there is)
    bot_input_ids = input_ids
    # generate a bot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=False,
        top_k=100,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
    #print the output
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return output



app = Flask(__name__)

# with open('file.txt','r') as file:
#     conversation = file.read()





@app.route("/")
def home(): 
	return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
# 	userText = request.args.get('msg')
# 	return str(bott.get_response(userText))

@app.route("/random", methods=['POST'])
def get_bot_response():
    a = list(map(str, request.form.values()))
    output = predictor(a[0])
    return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
	app.run(threaded=True, port=5000)