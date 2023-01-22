# Natural_Language_Understanding.py
import spacy

# load the spacy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    # process the text using spacy
    doc = nlp(text)

    # extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

def extract_noun_chunks(text):
    # process the text using spacy
    doc = nlp(text)

    # extract noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    return noun_chunks

def extract_verbs(text):
    # process the text using spacy
    doc = nlp(text)

    # extract verbs
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    return verbs
