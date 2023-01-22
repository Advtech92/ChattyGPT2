# Generate_Replies.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_replies(model, tokenizer, input_text):
    # encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # generate the reply
    reply_ids = model.generate(input_ids, max_length=200, do_sample=True, top_p=0.95, top_k=50)
    
    # decode the reply
    reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    return reply_text
