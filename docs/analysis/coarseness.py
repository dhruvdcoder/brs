# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from transformers import ElectraForPreTraining, ElectraTokenizerFast
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoModelForMaskedLM, AutoTokenizer, AutoModelForPreTraining
import torch
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

""
mlm_model_name = 'bert-base-uncased'
electra_model_name = 'google/electra-base-discriminator'

mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
electra_model = ElectraForPreTraining.from_pretrained(electra_model_name)
electra_tokenizer = ElectraTokenizerFast.from_pretrained(electra_model_name)


""
# Get the network and tokenizer
def electra_predict(masked_sentence: str, replacements: List[str])-> Tuple[List[str], List[float]]:
    outputs = []
    avgs = []
    for replacement in replacements:
        tokens = electra_tokenizer.tokenize(masked_sentence)
        mask_positions = [i for i, token in enumerate(tokens) if token == '[MASK]']
        assert len(mask_positions) ==1
        mask_position = mask_positions[0]+1 # add 1 to account for CLS
        sentence = masked_sentence.replace("[MASK]", replacement)
        encoded_input = electra_tokenizer.encode(sentence, return_tensors="pt")
        discriminator_outputs = torch.sigmoid(electra_model(encoded_input)[0].squeeze()).tolist()
        fake_or_not = discriminator_outputs[mask_position]
        del discriminator_outputs[mask_position]
        non_fake_avg = np.mean(discriminator_outputs)
        outputs.append(fake_or_not)
        avgs.append(non_fake_avg)
    assert len(replacements) == len(outputs)
    return replacements, outputs, avgs
        


""
def mlm_predict(sentence: str, topk: int=10, model_name: str='bert-base-uncased') -> Tuple[List[str], List[float]]:
    """
    Given a sentence with single [MASK], returns topk predictions with their respective probabilities
    """
    tokenizer = mlm_tokenizer
    model = mlm_model
    text = sentence
    encoded_input = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.tokenize(text)
    mask_positions = [i for i, token in enumerate(tokens) if token == '[MASK]']
    assert len(mask_positions) ==1
    mask_position = mask_positions[0]+1 # add 1 to account for CLS
    output = model(**encoded_input)
    scores, output_ids = torch.topk(torch.softmax(output.logits, dim=-1)[:,mask_position,:],k=topk,dim=-1)
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids.squeeze().tolist())
    scores = scores.squeeze().tolist()
    return output_tokens, scores


""
def plot_scores(scores: List[float], words: List[str], title: str, ylabel: str):


    x_pos = [i for i, _ in enumerate(words)]

    plt.bar(x_pos, scores, color='green')
    plt.xlabel("Words")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(x_pos, words, rotation='vertical')
    plt.show()

"""
# Manual exploration

Add sentences with \[MASK\] and a list of extra words (apart from top ranking predictions from MLM) to give to ELECTRA.
"""

masked_sentences = [
    "The [MASK] went into the hospital.",
    "[MASK] flew through the window."
]
extra_words_ = [
    ['car', 'ball', 'book', 'ship', 'play'],
    ['cat', 'plane', 'Jim'],
]

""
for i, (masked_sentence, extra_words) in enumerate(zip(masked_sentences, extra_words_)):
    print(f"Example {i+1}")
    print("================")
    print(f"Sentence: {masked_sentence}\n")
    mlm_predictions, mlm_scores =  mlm_predict(masked_sentence)
    print("Top 10 words from MLM and their probabilities.\n")
    for score, token in zip(mlm_scores, mlm_predictions):
        print(f"{token}: {score}")
    plot_scores(mlm_scores, mlm_predictions, "MLM probability for top MLM predictions", "model probability")
    electra_tokens, electra_scores, electra_avg_scores = electra_predict(masked_sentence, mlm_predictions + extra_words)
    print("\nCorrectness (score in [0,1] for the word / avg score for all words in the sentence) given by ELECTRA\n")
    for score, token, avg in zip(electra_scores, electra_tokens, electra_avg_scores):
        print(f"{token}: {(1-score):4f} / {(1-avg):4f}")
    plot_scores([1-s for s in electra_scores], electra_tokens, "ELECTRA scores for correctness in [0,1] for top MLM predictions", "model correctness score")
    print("\n")

"""
# References:

1. [ELECTRA on hugging face](https://huggingface.co/google/electra-base-discriminator)
"""

