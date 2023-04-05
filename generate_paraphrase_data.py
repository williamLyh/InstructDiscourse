import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm


def get_news_instruction_prompt(headline, stage_plan, sid, generated_list=[], prompt_version=1):
    stage_tags = ['main event', 'consequence', 'previous event', 'current context', 'historical event',
                                'future consequences', 'journalist evaluation', 'anecdotal event']
    stage_tags_code = ['<main event>', '<consequence>', '<previous event>', '<current context>', '<historical event>',
                                '<future consequences>', '<journalist evaluation>', '<anecdotal event>']
    
    discourse_definition = "The schema for discourse structure is defined below: "+\
                "<main event> refers to the major subject of the news article. "+\
                "<consequence> refers to an event or phenomenon that is caused by the main event. "+\
                "<previous event> refers to a specific event that occurred shortly before the main event. "+\
                "<current context> refers to the general context or worldstate immediately preceding the main event. "+\
                "<historical event> refers to an event occurring much earlier than the main event. "+\
                "<future consequences> refers to an analytical insight into future consequences or projections made by the journalist. "+\
                "<journalist evaluation> refers to a summary, opinion or comment made by the journalist. "+\
                "<anecdotal event> refers to anecdotal events that are uncertain and cannot be verified. The primary purpose is to provide more emotional resonance to the main event. \n\n"

    
    generated = ' '.join(generated_list)
    if prompt_version == 1:
        instruction_prompt = 'Continue writing a {} section for the below news article about {}: {}'.format(
                        stage_tags[stage_plan[sid]], headline, generated)
    
    elif prompt_version == 2:
        instruction_prompt = 'Continue writing a {} section for the below news article about {}: {}'.format(
                        stage_tags_code[stage_plan[sid]], headline, generated)
    
    elif prompt_version == 3:
        # Instruction version 3: with Code and explanation
        instruction_prompt = discourse_definition
        instruction_prompt += 'Continue writing a {} section for the below news article about {}: {}'.format(
                            stage_tags_code[stage_plan[sid]], headline, generated)
        
    elif prompt_version == 4:
    # Instruction version 4: with Code, explanation, only previous stage context
        instruction_prompt = discourse_definition
        if sid == 0:
            instruction_prompt += 'Writing a {} section for a news article about "{}".'.format(
                                stage_tags_code[stage_plan[sid]], headline)
        else:
            instruction_prompt += "The previous discourse structure is defined below: \n\n{}\n\n".format(
            ' '.join([stage_tags[tag] for tag in stage_plan[:sid]]))
            instruction_prompt += 'Continue writing a {} section for the news article about "{}".'.format(
                                stage_tags_code[stage_plan[sid]], headline)
    elif prompt_version == 5:
        # Instruction version 5: with Code, explanation, only previous stage context
        instruction_prompt = discourse_definition
        instruction_prompt += "The previous discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
        instruction_prompt += "The later discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[sid+1:]]))
        instruction_prompt += 'Write a {} section for the news article about "{}".'.format(
                            stage_tags_code[stage_plan[sid]], headline)
    
    return instruction_prompt


if __name__ == '__main__':
    step_data_path = '/mnt/nas_home/yl535/datasets/kaggle_all_the_news/preprocessed_step_level_data.json'
    step_data = json.load(open(step_data_path, 'r'))

    train_data = {key:val for key, val in step_data['train_data'].items()}
    valid_data = {key:val for key, val in step_data['valid_data'].items()}
    
    model_name = '/mnt/nas_home/yl535/InstructDiscourse/model-checkpoint/step-generator-flanT5-code-example/checkpoint-600000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()


    paraphrase_dataset = {}
    dataset = valid_data


    intermediate_savepoint = 150000
    resume_point = 0
    data_saving_path = '/mnt/nas_home/yl535/datasets/kaggle_all_the_news/'

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
            # generated_doc[dataset['id'][i]] = []
            # reference_doc[dataset['id'][i]] = []

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