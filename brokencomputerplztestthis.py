import discord
from discord.ext import commands
import nest_asyncio
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import numpy as np
import torch

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)

# Set up the Falcon-7B model
falcon_model_name = "tiiuae/falcon-7b"
falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)
falcon_model = AutoModelForCausalLM.from_pretrained(falcon_model_name, trust_remote_code=True).to('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the Llama model
llama_model = Llama("C:\\Users\\gray00\\hackerbotllamafalcon\\ggml-vicuna-7b-4bit-rev1.bin")

# Set up a dictionary to store conversation history
conversation_history = {}

def preprocess_input(user_input):
    # Convert the input into a matrix of vectors
    input_matrix = np.array([list(user_input)])
    return input_matrix

async def generate_falcon_response(prompt, max_length=1024):
    # Generate a response using the Falcon model
    inputs = falcon_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    falcon_outputs = falcon_model.generate(inputs, max_length=max_length, do_sample=True)
    falcon_response = falcon_tokenizer.decode(falcon_outputs[0])
    return falcon_response

async def llama_generate(prompt):
    # Generate a response using the Llama model
    llama_response = llama_model.generate(prompt)
    return llama_response

async def generate_response(user_input, user_id):
    # Preprocess the user input
    user_input = preprocess_input(user_input)

    # Add the user input to the conversation history
    if user_id in conversation_history:
        conversation_history[user_id] += f" {user_input}"
    else:
        conversation_history[user_id] = user_input
    # Generate the response based on the conversation history
    falcon_response = await generate_falcon_response(conversation_history[user_id])
    llama_response = await llama_generate(falcon_response)

    # Add the response to the conversation history
    conversation_history[user_id] += f" {llama_response}"

    return llama_response

async def generate_loop_response(user_input, num_loops, user_id):
    response = user_input
    for _ in range(num_loops):
        response = await generate_response(response, user_id)
    return response

async def send_chunks(ctx, response):
    await ctx.send(response[:2000])  # Discord has a limit of 2000 characters per message

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

@bot.command()
@commands.cooldown(1, 30, commands.BucketType.user)
async def trideque(ctx, *, user_input):
    try:
        await ctx.send('Processing your input, please wait...')
        response = await generate_response(user_input, ctx.author.id)
        await send_chunks(ctx, response)
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")

@bot.command()
@commands.cooldown(1, 30, commands.BucketType.user)
async def loop(ctx, num_loops: int, *, user_input):
    try:
        await ctx.send('Processing your input, please wait...')
        response = await generate_loop_response(user_input, num_loops, ctx.author.id)
        await send_chunks(ctx, response)
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        msg = 'This command is on a cooldown, please try again in {:.2f}s'.format(error.retry_after)
        await ctx.send(msg)
    else:
        raise error

nest_asyncio.apply()
bot.run('key')
