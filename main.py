import json
import torch
import time
import threading
from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, Y, RIGHT, END, StringVar, IntVar, Checkbutton
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

# Llama Model
llm = Llama(model_path="C:\\Users\\Shadow\\ggml-vicuna-7b-4bit\\ggml-vicuna-7b-4bit-rev1.bin")

def llama_generate(prompt, max_tokens=200):
    output = llm(prompt, max_tokens=max_tokens)
    return output['choices'][0]['text']

# GPT-Neo Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125m')

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = tokenizer.pad_token_id

# Falcon-7B Model
falcon_model_name = "tiiuae/falcon-7b"
falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)
falcon_model = AutoModelForCausalLM.from_pretrained(falcon_model_name).to(device)

# Generate Response
def generate_response(input_text, use_llama=True, use_gpt_neo=True, use_falcon=True):
    global context_history
    context_history.append(input_text)
    context = ' '.join(context_history[-5:])  # Use the last 5 turns as context

    response = ""
    if use_llama:
        # Generate response using Llama
        llama_response = llama_generate(context)
        response += f"Llama: {llama_response}\n"

    if use_gpt_neo:
        # Generate response using GPT-Neo
        inputs = tokenizer.encode(context, return_tensors='pt', truncation=True, max_length=512).to(device)
        attention_mask = inputs.ne(tokenizer.pad_token_id).float().to(device)
        gpt_neo_outputs = model.generate(inputs, max_length=2000, do_sample=True, attention_mask=attention_mask)
        gpt_neo_response = tokenizer.decode(gpt_neo_outputs[0])
        response += f"GPT-Neo: {gpt_neo_response}\n"

    if use_falcon:
        # Generate response using Falcon-7B
        inputs = falcon_tokenizer.encode(context, return_tensors='pt', truncation=True, max_length=512).to(device)
        falcon_outputs = falcon_model.generate(inputs, max_length=2000, do_sample=True)
        falcon_response = falcon_tokenizer.decode(falcon_outputs[0])
        response += f"Falcon-7B: {falcon_response}\n"

    context_history.append(response)
    return response

# GUI setup
root = Tk()
root.title("AI Conversation")
root.geometry("800x600")
# Context History
context_history = []

# Model Weights
llama_weight = IntVar(value=50)
gpt_neo_weight = IntVar(value=50)

# Dynamic Prompt
dynamic_prompt = StringVar(value="Let's discuss something interesting.")

Label(root, text="Enter input:").grid(row=0, column=0, sticky="W")

input_text = Entry(root, width=100)
input_text.grid(row=1, column=0)


output_text = Text(root, wrap="word", width=80, height=20)
output_text.grid(row=2, column=0, padx=10, pady=10, rowspan=6)

scrollbar = Scrollbar(root, command=output_text.yview)
scrollbar.grid(row=2, column=1, sticky="ns", rowspan=6)
output_text.config(yscrollcommand=scrollbar.set)

# Model selection checkboxes
use_llama = IntVar(value=1)
use_gpt_neo = IntVar(value=1)
use_falcon = IntVar(value=1)
Checkbutton(root, text="Use Llama", variable=use_llama).grid(row=8, column=0, sticky="W")
Checkbutton(root, text="Use GPT-Neo", variable=use_gpt_neo).grid(row=9, column=0, sticky="W")
Checkbutton(root, text="Use Falcon-7B", variable=use_falcon).grid(row=10, column=0, sticky="W")

# Dynamic prompt entry
Label(root, text="Dynamic Prompt:").grid(row=11, column=0, sticky="W")
dynamic_prompt_entry = Entry(root, textvariable=dynamic_prompt, width=100)
dynamic_prompt_entry.grid(row=12, column=0)

# Generate response and update GUI
def on_generate_click():
    user_input = input_text.get()
    response = generate_response(user_input, use_llama.get(), use_gpt_neo.get(), use_falcon.get())
    output_text.insert(END, f"You: {user_input}\n{response}\n")
    input_text.delete(0, END)  # Clear input field

# Generate response based on dynamic prompt
def on_dynamic_prompt_click():
    response = generate_response(dynamic_prompt.get(), use_llama.get(), use_gpt_neo.get(), use_falcon.get())
    output_text.insert(END, f"{response}\n")

# Bind enter key to button click event
def on_enter_key(event):
    on_generate_click()

Button(root, text="Generate", command=on_generate_click).grid(row=1, column=1)
Button(root, text="Use Dynamic Prompt", command=on_dynamic_prompt_click).grid(row=12, column=1)
root.bind('<Return>', on_enter_key)

root.mainloop()
