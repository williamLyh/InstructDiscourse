import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from dataclass import get_news_instruction_prompt

def generate_paraphrase_data():
    step_data_path = '/home/yinhong/Documents/datasets/kaggle_all_the_news/preprocessed_step_level_data.json'
    step_data = json.load(open(step_data_path, 'r'))

    train_data = {key:val for key, val in step_data['train_data'].items()}
    valid_data = {key:val for key, val in step_data['valid_data'].items()}
    
    model_name = '/home/yinhong/Documents/source/InstructDiscourse/model-checkpoint/step-generator-flanT5-code-example/checkpoint-800000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()


    paraphrase_dataset = {}
    dataset = valid_data


    intermediate_savepoint = 150000
    resume_point = 0
    data_saving_path = '/home/yinhong/Documents/datasets/kaggle_all_the_news/'

    evaluation_size = len(dataset['input_context'])
    pbar = tqdm(total=evaluation_size, initial=resume_point)
    for i in range(resume_point, evaluation_size):
        if dataset['id'][i] not in paraphrase_dataset:
            # if len(paraphrase_dataset)>=2:
            #     break
            paraphrase_dataset[dataset['id'][i]] = {'headline':None,
                                                    'stage_plan':None,
                                                    'generated_doc':[],
                                                    'reference_doc':[]
                                                    }

        instruction_prompt = get_news_instruction_prompt(dataset['headline'][i], 
                                                        dataset['stage_plan'][i], 
                                                        dataset['s_id'][i], 
                                                        generated_list=paraphrase_dataset[dataset['id'][i]]['generated_doc'], 
                                                        prompt_version=3
                                                        )
        # print(instruction_prompt)
        encoded_prompt = tokenizer(instruction_prompt, return_tensors="pt")

        outputs = model.generate(encoded_prompt['input_ids'].to(device),
                                attention_mask=encoded_prompt['attention_mask'].to(device),
                                max_length=128,
                                do_sample=True,
                                top_k=10,
                                early_stopping=True,
                                num_return_sequences=1,
                                )
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        paraphrase_dataset[dataset['id'][i]]['headline'] = dataset['headline'][i]
        paraphrase_dataset[dataset['id'][i]]['stage_plan'] = dataset['stage_plan'][i]
        paraphrase_dataset[dataset['id'][i]]['generated_doc'].append(decoded_output)
        paraphrase_dataset[dataset['id'][i]]['reference_doc'].append(dataset['target_text'][i])
        pbar.update(1)

        if (i+1) % intermediate_savepoint == 0:
            with open(data_saving_path+'paraphrase_dataset.json', 'w') as f:
                json.dump(paraphrase_dataset, f)
            f.close()
            print('Intermediate data saving at step {}.'.format(i+1))

def generate_inflation_paraphrase_data():
    data_saving_path = '/home/yinhong/Documents/datasets/kaggle_all_the_news/'
    step_data_path = '/home/yinhong/Documents/datasets/kaggle_all_the_news/preprocessed_step_level_data.json'
    step_data = json.load(open(step_data_path, 'r'))

    train_data = {key:val for key, val in step_data['train_data'].items()}
    valid_data = {key:val for key, val in step_data['valid_data'].items()}
    
    model_name = '/home/yinhong/Documents/source/InstructDiscourse/model-checkpoint/step-generator-flanT5-code-example/checkpoint-800000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda:1')
    model = model.to(device)
    model.eval()

    dataset = valid_data

    unique_doc_id = set()
    unique_doc_plan = []
    unique_doc_headline = []
    unique_doc_reference = []

    reference_text = []
    for idx, idx in enumerate(dataset['id']):
        if idx not in unique_doc_id:
            unique_doc_id.add(idx)
            unique_doc_plan.append(dataset['stage_plan'][idx])
            unique_doc_headline.append(dataset['headline'][idx])
            if reference_text!=[]:
                unique_doc_reference.append(reference_text)
            reference_text = []
        reference_text.append(dataset['target_text'][idx])
    unique_doc_reference.append(reference_text)

    paraphrase_dataset = {
        'id': unique_doc_id,
        'headline': unique_doc_headline,
        'stage_plan': unique_doc_plan,
        'reference': unique_doc_reference
    }

    generated_doc = []
    paraphrase_dataset_flat = {
        'id': [],
        'headline': [],
        'stage_plan': [],
        'reference': [],
        'generated': []
    }
    inflation = 3
    intermediate_savepoint = 10000
    resume_point = 0
    data_size = len(paraphrase_dataset['id'])
    pbar = tqdm(total=data_size*inflation, initial=resume_point)
    for idx in range(resume_point, data_size):
        for time in range(inflation):
            generated_text = []
            for sid in range(len(paraphrase_dataset['stage_plan'][idx])):
                instruction_prompt = get_news_instruction_prompt(
                    paraphrase_dataset['headline'][idx], 
                    paraphrase_dataset['stage_plan'][idx], 
                    sid,
                    generated_text, 
                    prompt_version=3)
                encoded_prompt = tokenizer(instruction_prompt, return_tensors='pt')
                outputs = model.generate(encoded_prompt['input_ids'].to(device),
                                        attention_mask=encoded_prompt['attention_mask'].to(device),
                                        max_length=128,
                                        do_sample=True,
                                        top_k=10,
                                        early_stopping=True,
                                        num_return_sequences=1,
                                        )
                generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                generated_text.append(generated)
            generated_doc.append(generated_text)
            paraphrase_dataset_flat['id'].append(idx)
            paraphrase_dataset_flat['headline'].append(paraphrase_dataset['headline'][idx])
            paraphrase_dataset_flat['stage_plan'].append(paraphrase_dataset['stage_plan'][idx])
            paraphrase_dataset_flat['reference'].append(paraphrase_dataset['reference'][idx])
            paraphrase_dataset_flat['generated'].append(generated_text)
            pbar.update(1)

            
        if (idx+1) % intermediate_savepoint == 0:
            with open(data_saving_path+'paraphrase_dataset_inflation.json', 'w') as f:
                json.dump(paraphrase_dataset_flat, f)
            f.close()
            print('Intermediate data saving at step {}.'.format(idx+1))
        
    with open(data_saving_path+'paraphrase_dataset_inflation.json', 'w') as f:
        json.dump(paraphrase_dataset_flat, f)
    f.close()
    print('Intermediate data saving at step {}.'.format(idx+1))


if __name__ == '__main__':
    generate_paraphrase_data()
    # generate_inflation_paraphrase_data()