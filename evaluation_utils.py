from collections import Counter
import numpy as np
# from dataset_preparation import automatic_stage_tagging_sentence_level
from tqdm import tqdm
import sys
import spacy
from rouge import Rouge 
from sacrebleu.metrics import BLEU, CHRF, TER
import re
import re
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def split_bins(start=0, end=0, bins=4):
    return tuple(np.ceil(np.linspace(start=start, stop=end, num=bins + 1)).astype(np.int32))

def extract_instruction(text, return_list=False):
    text =  text.split('<INSTR_START> ')[1].split('<INSTR_END>')[0].strip()
    stage_separation_tokens = ['<INSTR_NEXT>']
    text_list = None
    if return_list:
        pattern = '|'.join([token+' ' for token in stage_separation_tokens])
        text_list = re.split(pattern, text) if text != '' else []
    
    for token in stage_separation_tokens:
        text = text.replace(token, '')
    return text, text_list

def sentence_splitter(text_doc, flat=False):
    splitted_doc = []
    for doc in text_doc:
        ans = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)\
(?<=\.|\?|\!|\’|\”)(?<![0-9]\.)\s(?!\w\.)(?=[A-Z])', doc)
        if flat:
            splitted_doc += ans
        else:
            splitted_doc.append(ans)
    return splitted_doc

# Reshape the sentence stage label to document level
def reshape_stage_label_to_document_level(generated_stage_labels, reference_stage_labels, reference_doc):
    '''The generated_stage_labels and reference_stage_labels are in flatten shape. 
    This function reshape them to document level.
    The reference_doc has the document level structure that we can use to reshape the stage labels.
    '''
    generated_stage_label_doc = []
    reference_stage_label_doc = []

    global_list_idx = 0
    for doc_list in reference_doc.values():
        temp_generated_stage_label = []
        temp_reference_stage_label = []
        for sid, sent in enumerate(doc_list):
            temp_generated_stage_label.append(generated_stage_labels[global_list_idx])
            temp_reference_stage_label.append(reference_stage_labels[global_list_idx])
            global_list_idx += 1
        generated_stage_label_doc.append(temp_generated_stage_label)
        reference_stage_label_doc.append(temp_reference_stage_label)

    generated_stage_label_doc[0], reference_stage_label_doc[0]
    return generated_stage_label_doc, reference_stage_label_doc

##################################################################
## N-gram Auxiliary Functions
##################################################################
def exact_match(predict_seq, reference_seq):
    match_cnt = 0
    total_cnt = 0
    for predicted_plan, reference_plan in zip(predict_seq, reference_seq):
        for p1, p2 in zip(predicted_plan, reference_plan):
            if p1 and p2 and p1==p2:
                match_cnt += 1

        total_cnt += len(predicted_plan)
    return match_cnt/total_cnt

def plan_to_unigram(plan):
    return [(stage) for stage in plan]
    
def plan_to_bigram(plan):
    result = []
    for i in range(len(plan)-1):
        result.append(tuple(plan[i:i+2]))
    return result

def plan_to_trigram(plan):
    result = []
    for i in range(len(plan)-2):
        result.append(tuple(plan[i:i+3]))
    return result


def n_gram_match_rate(predict_seq, reference_seq, n=1):
    n_to_N_string = {1:'Unigram', 2:'Bigram', 3:'Trigram'}
    if n==1:
        reference_ngram = [plan_to_unigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_unigram(plan) for plan in predict_seq]

    elif n==2:
        reference_ngram = [plan_to_bigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_bigram(plan) for plan in predict_seq]

    elif n==3:
        reference_ngram = [plan_to_trigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_trigram(plan) for plan in predict_seq]
    else:
        print('Wrong n-gram number. ')

    average_match_rate = []
    for ngram1, ngram2 in zip(reference_ngram, prediction_ngram):
        ngram1_cnt = Counter(ngram1)
        ngram2_cnt = Counter(ngram2)
        match_cnt = 0
        for ngram in ngram2_cnt.keys():
            if ngram in ngram1_cnt:
                # print(bigram1_cnt[bigram],bigram2_cnt[bigram])
                # print(min(bigram1_cnt[bigram], bigram2_cnt[bigram]))
                match_cnt += min(ngram1_cnt[ngram], ngram2_cnt[ngram])
        if sum(ngram2_cnt.values()) != 0:
            match_rate = match_cnt / sum(ngram2_cnt.values())
            average_match_rate.append(match_rate)
    print('{} match rates'.format(n_to_N_string[n]), np.mean(average_match_rate))


#####################################################################################
# Recipe Auxiliary Functions
#####################################################################################
# predefined keywords for plan schema 
preprocessing_vocab = ['peel', 'cut', 'chop', 'drain', 'beat', 'spread', 'spread', 'keep', 'mince',
                       'crush', 'uncover', 'roll', 'slice', 'rinse', 'melt', 'preheat', 'portion',
                       'prepare', 'season', 'fold', 'sift', 'spray', 'swish', 'spoon', 'rub', 'marinate',
                       'press', 'mash', 'strain', 'fill', 'stir', 'soak', 'knead', 'prep',' punch', 'macerate',
                       'whip', 'start']
mixing_vocab = ['mix', 'add', 'combine', 'blend', 'saute', 'whisk', 'immerse']
moving_vocab = ['move', 'put', 'set', 'pour', 'place', 'transfer', 'arrange', 'scrape', 'scoop', 'jar']
cooking_vocab = ['fry', 'bake', 'cook', 'simmer', 'refrigerate', 'boil', 'brown', 'toss', 'heat', 'ferment',
                 'warm', 'flip', 'coat', 'stir-fry', 'grill', 'steam', 'toast']
postprocessing_vocab = ['cool', 'spread', 'unmold', 'keep', 'garnish', 'chill', 'top', 'turn', 'rinse',
                        'drain', 'melt', 'cover', 'reduce', 'discard', 'store', 'separate', 'sprinkle',
                        'remove', 'shake', 'lay', 'trim', 'taste', 'divid', 'drizzle', 'dip', 'frosting', 
                        'plate', 'form']
final_vocab = ['serve', 'make', 'yield', 'drink', 'enjoy', 'wrap', 'decor', 'decorate', 'final', 'finish']
keyword_vocab = preprocessing_vocab + mixing_vocab + moving_vocab + cooking_vocab + postprocessing_vocab + final_vocab
ignoring_words = ['cooked', 'baking']

def verb_to_label(w):
    # {'preprocessing_vocab':1, 'mixing_vocab':2, 'moving_vocab':3, 'cooking_vocab':4, 'postprocessing_vocab':5, 'final_vocab':6, 'unlabelled':0}
    label = 0
    if w in preprocessing_vocab:
        label = 1
    elif w in mixing_vocab:
        label = 2
    elif w in moving_vocab:
        label = 3
    elif w in cooking_vocab:
        label = 4
    elif w in postprocessing_vocab:
        label = 5
    elif w in final_vocab:
        label = 6
    return label

def labels_reduce(labels):
    # deciding which is the leading action, based on priority rank
    # sentence with no labels will be assign 5, the 'postprocessing' tag
    stage_to_idx = {'preprocessing':1, 'mixing':2, 'moving':3, 'cooking':4, 'postprocessing':5, 'final':6, 'general':0}
    labels = [label for label in labels if label!=0]
    if labels == []:
        return stage_to_idx['general']

    if stage_to_idx['cooking'] in labels:
        return stage_to_idx['cooking']
    
    return labels[-1]

def automatic_stage_tagging_sentence_level(instruction, spacy_tokenizer):
    # instruction = '; '.join(['I '+sent for sent in instruction.split('; ')])
    words = spacy_tokenizer('I '+instruction.lower())
    labels = []
    for word in words:
        if word.pos_ in ['VERB'] and word.text not in ignoring_words:
            if word.lemma_ == 'stir':
                if 'stir in' in words.text or 'stir together' in words.text:
                    labels.append(2)
                    continue
                if 'stir-fry' in words.text:
                    labels.append(4)
                    continue
            labels.append(verb_to_label(word.lemma_)) # verb not in pre-defined verb lists will be assigned 0. 
    label = labels_reduce(labels)
    return label


def compute_stage_matching(generation_doc_list, stage_reference_data):
    '''
    generation_doc_list and test_stage_data have format of list of list
    '''
    spacy_tokenizer = spacy.load("en_core_web_sm", disable=['parser', 'senter', 'ner'])
    scores = []
    for generated_text_list, teat_stage in tqdm(zip(generation_doc_list, stage_reference_data), 
                                                total=len(generation_doc_list)):
        labels = []
        for sent in generated_text_list:
            # words = spacy_tokenizer(sent)
            label = automatic_stage_tagging_sentence_level(sent, spacy_tokenizer)
            labels.append(label)
        
        match_cnt = 0.0
        for generated_label, reference_label in zip(labels, teat_stage):
            if generated_label == reference_label:
                match_cnt += 1
        scores.append(match_cnt/len(teat_stage))
    return np.average(scores)


##################################################################
## Visualization Functions
##################################################################
def draw_bigram_heatmap(stage_plan, stage_tag_mapping):
    bigrams = [plan_to_bigram(plan) for plan in stage_plan]
    bigrams = [bi for bigs in bigrams for bi in bigs]
    bigram_counter = Counter(bigrams)

    total_count = bigram_counter.total()
    heat_counter = np.zeros((len(stage_tag_mapping),len(stage_tag_mapping)))
    total_count = bigram_counter.total()
    for t1, id1 in stage_tag_mapping.items():
        for t2, id2 in stage_tag_mapping.items():
            heat_counter[id1, id2] = bigram_counter[(id1,id2)]/total_count
    print("Total bi-gram count {}".format(total_count))
    plt.imshow(heat_counter)
    plt.yticks(np.arange(len(stage_tag_mapping)), labels=stage_tag_mapping.keys())
    plt.xticks(np.arange(len(stage_tag_mapping)), labels=stage_tag_mapping.keys(), rotation=45, ha='right', rotation_mode='anchor')

    plt.show()


##################################################################
## Ingredient coverage & extra
##################################################################
def remove_substring_ingr(ingr_list):
    ingredient_list = sorted(ingr_list,key=len)   
    ingr_to_remove = []
    for i in range(len(ingr_list)):
        for j in range(i+1, len(ingr_list)):
            if ingr_list[i] in ingr_list[j]:
                ingr_to_remove.append(ingr_list[i])
                break
                
    for ingr in ingr_to_remove:
        ingr_list.remove(ingr)
    return ingr_list

def calculate_ingredient_coverage_and_hallucination(input_text, generated_text, ingredient_dir):
    with open(ingredient_dir, "r") as fp:
        ingredient_set = json.load(fp)

    # Calculate coverage 
    coverage_percentage_buffer = []
    hallucination_percentage_buffer = []
    for input_line, generation_line in zip(input_text, generated_text):
        # extract ingredients from input textual gredient
        ingredient_list = []
        for ingr in ingredient_set:
            if ingr in input_line.split('Ingredients:')[1].split('Instructions:')[0]:
                ingredient_list.append(ingr)
        ingredient_list = remove_substring_ingr(ingredient_list)

        # count the covered ingredients
        coverage_cnt = 0
        for ingr in ingredient_list:
            if ingr in generation_line:
                coverage_cnt += 1
        
        coverage = coverage_cnt/len(ingredient_list) if len(ingredient_list)!=0 else 0
        coverage_percentage_buffer.append(coverage)
        
        # count the hallucinated ingredients
        hallucination_list = []
        for ingr in ingredient_set:
            if (ingr not in ingredient_list ) and (ingr in generation_line):
                hallucination_list.append(ingr)

        hallucination_list = remove_substring_ingr(hallucination_list)

        # remove convered ingredient substring
        ingr_to_remove = set()
        for ingr in hallucination_list:
            for ingr2 in ingredient_list:
                if ingr in ingr2:
                    ingr_to_remove.add(ingr)
        [hallucination_list.remove(ingr) for ingr in ingr_to_remove]


        hallucination = len(hallucination_list)/len(ingredient_list) if len(ingredient_list)!=0 else 1
        hallucination_percentage_buffer.append(hallucination)

    coverage_percentage = np.average(coverage_percentage_buffer)
    hallucination_percentage = np.average(hallucination_percentage_buffer)
    print('Coverage: {}. Hallucination: {}'.format(coverage_percentage, hallucination_percentage))



##################################################################
## Surface Fluency
##################################################################
def evaluate_fluency(predicted_text, reference_text):
    '''Compute BLEU and Rouge score'''

    # Bleu
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predicted_text, [reference_text])
    print(bleu_score)

    # Rouge
    rouge = Rouge()
    rouge_score = rouge.get_scores(predicted_text, reference_text)
    rouge_score = np.mean([case['rouge-l']['f'] for case in rouge_score])
    print('Rouge-L Score: {}'.format(rouge_score))

    # Meteor
    

##################################################################
## Stage Accuracy
##################################################################

def evaluate_stage_accuracy(predicted_text, reference_text, stage_classifier_path, batch_size=32):
    '''Compute stage accuracy score
    predicted_text: list of generated text (list of string)
    reference_text: list of reference text (list of string)
    stage_classifier_path: path to the stage classifier model
    '''
    class BertStageClassifierInferenceDataset(Dataset):
        def __init__(self, data, tokenizer=None, max_length=256):
            self.text = data
            # self.tag = data['tag']
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.text)

        def __getitem__(self, idx):        
            encoding_text = self.tokenizer(self.text[idx],
                                    max_length=self.max_length,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='pt')#.input_ids
            encoding = {'input_ids': encoding_text['input_ids'].squeeze(0)}
            encoding['attention_mask'] = encoding_text['attention_mask'].squeeze(0)
            return encoding


    bert_tokenizer = AutoTokenizer.from_pretrained(stage_classifier_path)
    model = AutoModelForSequenceClassification.from_pretrained(stage_classifier_path)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    prediction_dataset = BertStageClassifierInferenceDataset(predicted_text,
                                            bert_tokenizer, max_length=256)
    reference_dataset = BertStageClassifierInferenceDataset(reference_text,
                                            bert_tokenizer, max_length=256)
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size)
    reference_loader = DataLoader(reference_dataset, batch_size=batch_size)


    valid_stage_labels = []
    for batch in tqdm(prediction_loader):
        outputs = model(batch['input_ids'].to(device),
                        attention_mask = batch['attention_mask'].to(device))
        class_score = torch.softmax(outputs.logits,dim=-1)
        class_prediction = torch.argmax(class_score,dim=-1)
        valid_stage_labels += class_prediction.tolist()

    reference_stage_labels = []
    for batch in tqdm(reference_loader):
        outputs = model(batch['input_ids'].to(device),
                        attention_mask = batch['attention_mask'].to(device))
        class_score = torch.softmax(outputs.logits,dim=-1)
        class_prediction = torch.argmax(class_score,dim=-1)
        reference_stage_labels += class_prediction.tolist()

    print('Stage accuracy: ', exact_match([valid_stage_labels], [reference_stage_labels]))
    return valid_stage_labels, reference_stage_labels

def calculate_positional_accuracy(generated_stage_label_doc, reference_stage_label_doc, num_bins_defaule=10):
    bin_match_rate_global = []
    for generated_stage_label, reference_stage_label in zip(generated_stage_label_doc, reference_stage_label_doc):
        num_bins = min(num_bins_defaule, len(generated_stage_label))
        plan_bin = split_bins(start=0, end=len(generated_stage_label), bins=num_bins)
        bin_gaps = [plan_bin[i+1] - plan_bin[i] for i in range(num_bins)]
        bin_match_cnt = [0]*num_bins
        for i in range(len(generated_stage_label)):
            bin_idx = np.digitize(i, plan_bin) - 1
            if generated_stage_label[i] == reference_stage_label[i]:
                bin_match_cnt[bin_idx] += 1
        bin_match_rate = [cnt/bin_gaps[i] for i, cnt in enumerate(bin_match_cnt)]
        bin_match_rate_global.append(bin_match_rate)

    positional_accuracy = []
    for i in range(num_bins_defaule):
        temp_match_rate_list = []
        for line in bin_match_rate_global:
            if i<len(line):
                temp_match_rate_list.append(line[i])
        positional_accuracy.append(np.average(temp_match_rate_list))
    print(positional_accuracy)
    print('Positional accuracy: ', np.average(positional_accuracy))