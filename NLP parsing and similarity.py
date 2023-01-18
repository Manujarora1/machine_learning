import spacy
import allennlp
from allennlp.predictors.predictor import Predictor
import re
from nltk import tokenize
from nltk.tree import Tree

def install_dependencies():
    !pip install allennlp==0.9.0
    !pip install overrides==4.1.2
    !python -m spacy download en_core_web_sm

def load_model():
    global nlp
    nlp = spacy.load("en_core_web_sm")
    global predictor
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

def parse_sentence(sentence):
    sentence = sentence.rstrip('?:!.,;')
    parser_output = predictor.predict(sentence=sentence)
    tree_string = parser_output["trees"]
    return tree_string

def get_tree(tree_string):
    tree = Tree.fromstring(tree_string)
    return tree

def get_flattened(t):
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final

def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return last_NP,last_VP
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)

def get_termination_portion(main_string, sub_string):
    combined_sub_string = sub_string.replace(" ", "")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ", "")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
    return None

def process_sentence(sentence):
    tree_string = parse_sentence(sentence)
    tree = get_tree(tree_string)
    last_nounphrase, last_verbphrase = get_right_most_VP_or_NP(tree)
    last_nounphrase_flattened = get_flattened(last_nounphrase)




def get_first_VP_and_NP_and_sentence(parse_tree, first_NP=None, first_VP=None, first_sent=None):
    if len(parse_tree.leaves()) == 1:
        return get_flattened(first_NP), get_flattened(first_VP), get_flattened(first_sent)
    last_subtree = parse_tree[-1]

    if last_subtree.label() == "NP" and not first_NP:
        first_NP = last_subtree
    elif last_subtree.label() == "VP" and not first_VP:
        first_VP = last_subtree
    elif last_subtree.label() == "S" and not first_sent:
        first_sent = last_subtree

    return get_first_VP_and_NP_and_sentence(last_subtree, first_NP, first_VP, first_sent)



!pip install transformers==2.8.0

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# GPT2tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# GPT2model = TFGPT2LMHeadModel.from_pretrained("distilgpt2",pad_token_id=GPT2tokenizer.eos_token_id)
GPT2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
GPT2model = TFGPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=GPT2tokenizer.eos_token_id)

partial_sentence = "The old woman was sitting under a tree and"
input_ids = GPT2tokenizer.encode(partial_sentence,return_tensors='tf')
print (input_ids)
maximum_length = len(partial_sentence.split())+40

# Activate top_k sampling and top_p sampling with only from 90% most likely words
sample_outputs = GPT2model.generate(
    input_ids, 
    do_sample=True, 
    max_length=maximum_length, 
    top_p=0.80, # 0.85 
    top_k=30,   #30
    repetition_penalty  = 10.0,
    num_return_sequences=10
)

import nltk
nltk.download('punkt')
from nltk import tokenize
generated_sentences=[]

for i, sample_output in enumerate(sample_outputs):
    decoded_sentence = GPT2tokenizer.decode(sample_output, skip_special_tokens=True)
    # final_sentence = decoded_sentence
    final_sentence = tokenize.sent_tokenize(decoded_sentence)[0]
    generated_sentences.append(final_sentence)
    print (i,": ",final_sentence)

!pip install sentence-transformers==1.0.0

from sentence_transformers import SentenceTransformer, util
BERT_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

possible_false_sentences = ["The old woman was sitting under a tree and there were four men, each wearing dark clothing.",
                            "The old woman was sitting under a tree and had to look at her face.",
                            "The old woman was sitting under a tree and talking to the other lady, but when she noticed that I could hear myself in front of her with my voice it had gone up.",
                            "The old woman was sitting under a tree and saw him in the middle of nowhere.",
                            "The old woman was sitting under a tree and looking around at the bushes.",
                            "The old woman was sitting under a tree and gulping cocktail.",
                            "The old woman was sitting under a tree and staring at her, making the decision to leave.",
                            "The old woman was sitting under a tree and the man in his thirties came around her with some scissors, sawing it all as if she'd eaten him for lunch.",
                            "The old woman was sitting under a tree and staring at the window.",
                            "The old woman was sitting under a tree and she said, 'You know what?'",
                            "The old woman was sitting under a tree and staring at her like she would become the next, or whatever it looked.",
                            "The old woman was sitting under a tree and drinking tea."]




false_sentences_embeddings = BERT_model.encode(possible_false_sentences)
original_sentence_embedding = BERT_model.encode([original_sentence])

import scipy
distances = scipy.spatial.distance.cdist(original_sentence_embedding, false_sentences_embeddings, "cosine")[0]
print (distances)

results = zip(range(len(distances)), distances)
results = sorted(results, key=lambda x: x[1])
print (results)

dissimilar_sentences =[]
for idx, distance in results:
  dissimilar_sentences.append(possible_false_sentences[idx])
  print (possible_false_sentences[idx])

false_sentences_list_final = reversed(dissimilar_sentences)
for sent in false_sentences_list_final:
  print (sent)
