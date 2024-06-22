import torch
import time
import nltk
from nltk.tokenize import sent_tokenize
from googletrans import Translator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
import sacrebleu
from rouge_score import rouge_scorer

nltk.download('punkt')

# Loading Model
model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Loading text file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Save text file
def save_text(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


# Setting up pipeline
nlp = pipeline('text2text-generation', model=model, tokenizer=tokenizer, truncation=True)

# Paraphrasing using LLM
def llm_paraphrase(text):
    # tokenization
    sentences = nltk.sent_tokenize(text)

    # paraphrasing
    generated_txt = []
    generated_txt2 = []
    paraphrased_text = ""
    for i in range(len(sentences)):
        generated_txt.append(nlp(sentences[i]))

    for i in range(len(generated_txt)):
        generated_txt2.append(generated_txt[i][0]['generated_text'])

    # join sentences
    paraphrased_text = " ".join(generated_txt2)
    
    save_text(paraphrased_text, "res/llm_paraphrased.txt")

    return paraphrased_text

# Custom Paraphraser

def custom_paraphrase(text, src_lang='en', mid_lang="fr"):
    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=mid_lang).text
    back_translated = translator.translate(translated, src=mid_lang, dest=src_lang).text
    save_text(back_translated, "res/custom_paraphrased.txt")
    return back_translated


# Evaluation

def evaluate(text):
    # LLM Paraphrasing
    start_time = time.time()
    llm_paraphrased = llm_paraphrase(text)
    llm_time = time.time() - start_time
    

    # Custom Paraphrasing
    start_time = time.time()
    custom_paraphrased = custom_paraphrase(text)
    custom_time = time.time() - start_time

    # BLEU Score
    bleu_score_llm = sacrebleu.corpus_bleu([llm_paraphrased], [text]).score
    bleu_score_custom = sacrebleu.corpus_bleu([custom_paraphrased], [text]).score

    # ROUGE Score
    rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_llm = rouge_scorer_.score(text, llm_paraphrased)
    rouge_scores_custom = rouge_scorer_.score(text, custom_paraphrased)

    return {
        'llm': {
            'paraphrased_text': llm_paraphrased,
            'time': llm_time,
            'bleu': bleu_score_llm,
            'rouge': rouge_scores_llm,
        },
        'custom': {
            'paraphrased_text': custom_paraphrased,
            'time': custom_time,
            'bleu': bleu_score_custom,
            'rouge': rouge_scores_custom,
        }
    }

file_path = 'sample.txt'
text = load_text(file_path)
evaluation_results = evaluate(text)

print("LLM Method Results:")
print(f"Time: {evaluation_results['llm']['time']} seconds")
print(f"BLEU Score: {evaluation_results['llm']['bleu']}")
print(f"ROUGE Scores: {evaluation_results['llm']['rouge']}\n")

print("Custom Method Results:")
print(f"Time: {evaluation_results['custom']['time']} seconds")
print(f"BLEU Score: {evaluation_results['custom']['bleu']}")
print(f"ROUGE Scores: {evaluation_results['custom']['rouge']}")






