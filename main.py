# Main.py
import discord
import traceback
import asyncio
from discord.client import Client
from fine_tune import fine_tune
from generate_replies import generate_replies
from deep_learning import deep_learning
from database import connect_to_database
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
DB_FILE = os.getenv("DB_FILE")
EMOTION_DATASET = os.getenv("EMOTION_DATASET")
# Get the user id of the user who will receive the console output
OWNER_ID = os.getenv("OWNER_ID")

async def restart():
    await client.close()
    await asyncio.create_subprocess_exec("python3","main.py")
    await asyncio.sleep(1)

# Create the Discord client
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
client = Client(intents=intents)

@client.event
async def on_ready():
    print("Bot is ready!")


async def fine_tune_async(model, tokenizer, EMOTION_DATASET, conn):
    # Fine-tune the model and get the tokenizer
    model, tokenizer = fine_tune(model, tokenizer, EMOTION_DATASET)

    # Train the model on previous conversations
    deep_learning(model, tokenizer, conn)

@client.event
async def on_message(message):
    print(f"Received Message: {message.content} from {message.author} in {message.channel}")
    model = AutoModelWithLMHead.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if message.content.startswith("!train"):
        print(f'Received !train command from {message.author}')
        try:
            # Connect to the database
            conn = connect_to_database(DB_FILE)

            await message.channel.send("Training started, please wait...")

            # Create a new task to perform the training process asynchronously
            task = asyncio.create_task(fine_tune_async(model, tokenizer, EMOTION_DATASET, conn))

            # Wait for the task to complete
            await task
            await message.channel.send("Training complete!")
            print(f'Training complete')
        except Exception as e:
            tb = traceback.format_exc()
            owner = client.get_user(int(OWNER_ID))
            await owner.send(f"An error occurred: {e}\n\n{tb}")
    elif message.content.startswith("!reply "):
        print(f'Received !reply command from {message.author}')
        try:
            # Get the input text
            input_text = message.content[len("!reply "):]

            # Connect to the database
            conn = connect_to_database(DB_FILE)

            # Generate a reply
            reply_text = generate_replies(model, tokenizer, input_text)

            # Save the conversation to the database
            c = conn.cursor()
            c.execute("INSERT INTO conversation (input_text, reply_text) VALUES (?, ?)", (input_text, reply_text))
            conn.commit()

            await message.channel.send(reply_text)
        except Exception as e:
            tb = traceback.format_exc()
            owner = client.get_user(int(OWNER_ID))
            await owner.send(f"An error occurred: {e}\n\n{tb}")
    elif message.content.startswith("!restart"):
        await message.channel.send("Rebooting! Be right back!")
        await restart()

client.run(TOKEN)