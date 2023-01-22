#Deep_Learning.py
import torch
import pandas as pd
import sqlite3

def deep_learning(model, tokenizer, conn):
    c = conn.cursor()
    try:
        c.execute("SELECT input_text, reply_text FROM conversation")
        input_reply_pairs = c.fetchall()
        # load the emotion dataset
        df = pd.read_csv("datasets/emotion-emotion_69k.csv")
        situations = df["Situation"].tolist()
        emotions = df["emotion"].tolist()
        empathetic_dialogues = df["empathetic_dialogues"].tolist()
        labels = df["labels"].tolist()
        
        def data_generator():
                for input_text, reply_text, situation, emotion, empathetic_dialogue, label in zip(input_reply_pairs, situations, emotions, empathetic_dialogues, labels):
                    input_ids = tokenizer.encode(input_text, return_tensors="pt")
                    reply_ids = tokenizer.encode(reply_text, return_tensors="pt")
                    situation_ids = tokenizer.encode(situation, return_tensors="pt")
                    emotion_ids = tokenizer.encode(emotion, return_tensors="pt")
                    empathetic_dialogue_ids = tokenizer.encode(empathetic_dialogue, return_tensors="pt")
                    label_ids = tokenizer.encode(label, return_tensors="pt")
                    yield input_ids, reply_ids, situation_ids, emotion_ids, empathetic_dialogue_ids, label_ids
             # train the model on the data generator
        model.fit(data_generator())
        model.save_pretrained('model/')
    except sqlite3.OperationalError as e:
                print(f"Error: {e}")
                print("The table conversation does not exist")
