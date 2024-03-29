from torch.utils.data import Dataset, DataLoader

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
    discourse_definition_list = ["<main event> refers to the major subject of the news article.",
                "<consequence> refers to an event or phenomenon that is caused by the main event.",
                "<previous event> refers to a specific event that occurred shortly before the main event.",
                "<current context> refers to the general context or worldstate immediately preceding the main event.",
                "<historical event> refers to an event occurring much earlier than the main event.",
                "<future consequences> refers to an analytical insight into future consequences or projections made by the journalist.",
                "<journalist evaluation> refers to a summary, opinion or comment made by the journalist.",
                "<anecdotal event> refers to anecdotal events that are uncertain and cannot be verified. The primary purpose is to provide more emotional resonance to the main event."]

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
            instruction_prompt += 'Write a {} section for a news article about "{}".'.format(
                                stage_tags_code[stage_plan[sid]], headline)
        else:
            instruction_prompt += "The previous discourse structure is defined below: \n\n{}\n\n".format(
            ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
            instruction_prompt += 'Continue writing a {} section for the news article about "{}".'.format(
                                stage_tags_code[stage_plan[sid]], headline)
    elif prompt_version == 5:
        # Instruction version 5: with Code, explanation, previous and future stage context
        instruction_prompt = discourse_definition
        instruction_prompt += "The previous discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
        instruction_prompt += "The later discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[sid+1:]]))
        instruction_prompt += 'Write a {} section for the news article about "{}".'.format(
                            stage_tags_code[stage_plan[sid]], headline)
    
    elif prompt_version == 6:
    # Instruction version 6: with Code, explanation, only previous stage context
        instruction_prompt = discourse_definition
        capped_generated = generated.split(' ')[-200:]
        capped_generated = ' '.join(capped_generated)
        if sid == 0:
            instruction_prompt += 'Write a {} section for a news article about "{}".\n\n'.format(
                                stage_tags_code[stage_plan[sid]], headline)
        else:
            instruction_prompt += "The previous discourse structure is defined below: {}\n\n".format(
            ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
            instruction_prompt += 'Continue writing a {} section for the news article about "{}": {}.'.format(
                                stage_tags_code[stage_plan[sid]], headline, capped_generated)
            
    elif prompt_version == 7:
        # Instruction version 7: with Code, explanation, previous and future stage context, with generated text
        instruction_prompt = discourse_definition
        capped_generated = generated.split(' ')[-200:]
        capped_generated = ' '.join(capped_generated)
        instruction_prompt += "The previous discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
        instruction_prompt += "The later discourse structure of is defined below: \n\n{}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[sid+1:]]))
        instruction_prompt += 'Continue writing a {} section for the news article about "{}": {}.'.format(
                                stage_tags_code[stage_plan[sid]], headline, capped_generated)
        
    elif prompt_version == 8:
        # Instruction version 8: with Code, explanation, previous and future stage context, with generated text
        capped_generated = generated.split(' ')[-200:]
        capped_generated = ' '.join(capped_generated)
        instruction_prompt = 'Write a {} section for the news article about "{}". {}\n\n'.format(
                                stage_tags_code[stage_plan[sid]], headline, discourse_definition_list[stage_plan[sid]])
        instruction_prompt += "The previous discourse structure of is defined below: {}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[:sid]]))
        instruction_prompt += "The later discourse structure of is defined below: {}\n\n".format(
        ' '.join([stage_tags_code[tag] for tag in stage_plan[sid+1:]]))
        instruction_prompt += 'The previous generated news is: {} '.format(capped_generated)
    return instruction_prompt


class T5StepLevelNewsTrainingDataset(Dataset):
    def __init__(self, data, stage_tags=None, tokenizer=None, prompt_version=1):
        self.headline = data['headline']
        self.input_context = data['input_context']
        self.target_text = data['target_text']
        self.stage_label = data['stage_label']
        self.stage_plan = data['stage_plan']
        self.sid = data['s_id']
        self.prompt_version = prompt_version
        # self.tags = data['stage_plan']
        self.tokenizer = tokenizer
        if stage_tags:
            self.stage_tags = stage_tags
        else:
            self.stage_tags = ['<main event>', '<consequence>', '<previous event>', '<current context>', 
                               '<historical event>', '<future consequences>', '<journalist evaluation>', 
                               '<anecdotal event>']
        self.discourse_definiton = "The schema for discourse structure is defined below: "+\
                    "<main event> refers to the major subject of the news article. "+\
                    "<consequence> refers to an event or phenomenon that is caused by the main event. "+\
                    "<previous event> refers to a specific event that occurred shortly before the main event. "+\
                    "<current context> refers to the general context or worldstate immediately preceding the main event. "+\
                    "<historical event> refers to an event occurring much earlier than the main event. "+\
                    "<future consequences> refers to an analytical insight into future consequences or projections made by the journalist. "+\
                    "<journalist evaluation> refers to a summary, opinion or comment made by the journalist. "+\
                    "<anecdotal event> refers to anecdotal events that are uncertain and cannot be verified. The primary purpose is to provide more emotional resonance to the main event. \n\n"


    def __len__(self):
        return len(self.headline)

    def __getitem__(self, idx):    
        instruction_prompt = get_news_instruction_prompt(self.headline[idx], 
                                self.stage_plan[idx], 
                                self.sid[idx], 
                                [self.input_context[idx]], 
                                prompt_version=self.prompt_version 
                                )

        # self.stage_label[idx]
        return {'instruction': instruction_prompt,
                'target_text': self.target_text[idx],
        }
    

class T5StepLevelNewsInferenceDataset(Dataset):
    def __init__(self, data, stage_tags=None, tokenizer=None):
        self.headline = data['headline']
        self.input_context = data['input_context']
        self.target_text = data['target_text']
        self.stage_label = data['stage_label']
        # self.tags = data['stage_plan']
        self.tokenizer = tokenizer
        if stage_tags:
            self.stage_tags = stage_tags
        else:
            self.stage_tags = ['main event', 'consequence', 'previous event', 'current context', 'historical event',
                                'funture consequences', 'journalist evaluation', 'anecdotal event']

    def __len__(self):
        return len(self.headline)

    def __getitem__(self, idx):        
        instruction_prompt = 'Continue writing a {} section for the below news article about {}: {}'.format(
                            self.stage_tags[self.stage_label[idx]], 
                            self.headline[idx],
                            self.input_context[idx]
                            )
        return {'instruction': instruction_prompt,
                'target_text': self.target_text[idx],
        }
    


class T5StepLevelRecipeTrainingDataset(Dataset):
    def __init__(self, data, stage_tags=None, tokenizer=None, prompt_version=1):
        self.title = data['title']
        self.ingredients = data['ingredients']

        self.input_context = data['input_context']
        self.target_text = data['target_text']

        self.stage_plan = data['stage_plan']
        self.sid = data['sid']
        # self.tags = data['stage_plan']

        self.instruction_prompt = prompt_version

        self.tokenizer = tokenizer
        if stage_tags:
            self.stage_tags = stage_tags
        else:
            self.stage_tags = ['<general>', '<preprocessing>', '<mixing>', '<transferring>', '<cooking>', '<postprocessing>', '<final>']
            

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):        
        if self.instruction_prompt == 1:
            # Instruction version 1: Current stage + Previous context
            instruction_prompt = "Generate a single {} recipe instruction for {} using the following ingredients: {}.".format(
                self.stage_tags[self.stage_plan[idx][self.sid[idx]]],
                self.title[idx],
                ', '.join(self.ingredients[idx])
            )
            instruction_prompt += " The step instruction should be clear and concise, and include any necessary measurements or details." +\
                " If the step instruction requires a specific cooking technique, please also include how to perform that technique.\n\n"
            
            instruction_prompt += "\n".join(self.input_context[idx])
        
        elif self.instruction_prompt == 2:
            # Instruction version 2: Current stage + Definition of discourse structure + Previous context 
            instruction_prompt = "The tags of discourse structure is defined below: "+\
                    "<preprocessing> involves preparing the ingredients and equipment and preheating the oven or stove. "+\
                    "<mixing> involves combining the ingredients together to form a homogeneous mixture. The techniques include stirring, beating, whisking, kneading and folding. "+\
                    "<transferring> involves moving the mixture or dish from one location to another. The common transferring techniques include pouring, scooping and straining. "+\
                    "<cooking> involves applying heat to the mixture or dish to cook or bake it to completion. The common cooking techniques include baking, boiling, frying, grilling and roasting. "+\
                    "<postprocessing> refers to the steps taken after the cooking stage to finalise and present the finished dish. The common postprocessing techniques include cooling, garnishing, plating, serving and storing. "+\
                    "<final> refers to serving and consumption stage. "+\
                    "<general> refers to comments or side notes. \n\n"

            instruction_prompt += "Generate a single {} recipe instruction for {} using the following ingredients: {}.".format(
                self.stage_tags[self.stage_plan[idx][self.sid[idx]]],
                self.title[idx],
                ', '.join(self.ingredients[idx])
            )
            instruction_prompt += " The step instruction should be clear and concise, and include any necessary measurements or details." +\
                " If the step instruction requires a specific cooking technique, please also include how to perform that technique.\n\n"
            
            instruction_prompt += "\n".join(self.input_context[idx])


        return {'instruction': instruction_prompt,
                'target_text': self.target_text[idx],
        }
    


# class T5ParaphraseNewsTrainingDataset(Dataset):
#     def __init__(self, data, stage_tags=None, tokenizer=None, prompt_version=1):
#         self.headline = [datapoint['headline'] for datapoint in data]
#         self.input_text = [datapoint['input_doc'] for datapoint in data]
#         self.target_text = [datapoint['reference_doc'] for datapoint in data]
#         self.stage_plan = [datapoint['stage_plan'] for datapoint in data]
#         self.prompt_version = prompt_version
#         self.tokenizer = tokenizer
#         if stage_tags:
#             self.stage_tags = stage_tags
#         else:
#             self.stage_tags = ['<main event>', '<consequence>', '<previous event>', '<current context>', 
#                                '<historical event>', '<future consequences>', '<journalist evaluation>', 
#                                '<anecdotal event>']
#         self.discourse_definiton = "The schema for discourse structure is defined below: "+\
#                     "<main event> refers to the major subject of the news article. "+\
#                     "<consequence> refers to an event or phenomenon that is caused by the main event. "+\
#                     "<previous event> refers to a specific event that occurred shortly before the main event. "+\
#                     "<current context> refers to the general context or worldstate immediately preceding the main event. "+\
#                     "<historical event> refers to an event occurring much earlier than the main event. "+\
#                     "<future consequences> refers to an analytical insight into future consequences or projections made by the journalist. "+\
#                     "<journalist evaluation> refers to a summary, opinion or comment made by the journalist. "+\
#                     "<anecdotal event> refers to anecdotal events that are uncertain and cannot be verified. The primary purpose is to provide more emotional resonance to the main event. \n\n"

#     def __len__(self):
#         return len(self.headline)

#     def __getitem__(self, idx):    
#         if self.prompt_version == 1:
#             instruction_prompt = "Polish the following news article about {} to make it more fluent: {}".format(
#                     self.headline[idx],
#                     self.input_text[idx]
#             )

#         elif self.prompt_version == 2:
#             instruction_prompt = 'Continue writing a {} section for the below news article about {}: {}'.format(
#                     self.stage_tags[self.stage_label[idx]], 
#                     self.headline[idx],
#                     self.input_context[idx]
#                     )

        
#         # self.stage_label[idx]
#         return {'instruction': instruction_prompt,
#                 'target_text': self.target_text[idx],
#         }
    
class T5ParaphraseNewsDataset(Dataset):
    def __init__(self, data, stage_tags=None, tokenizer=None, prompt_version=1):
        self.headline = [datapoint['headline'] for datapoint in data]
        self.input_text = [datapoint['input_doc'] for datapoint in data]
        self.target_text = [datapoint['reference_doc'] for datapoint in data]
        self.stage_plan = [datapoint['stage_plan'] for datapoint in data]
        self.prompt_version = prompt_version
        self.tokenizer = tokenizer
        if stage_tags:
            self.stage_tags = stage_tags
        else:
            self.stage_tags = ['<main event>', '<consequence>', '<previous event>', '<current context>', 
                               '<historical event>', '<future consequences>', '<journalist evaluation>', 
                               '<anecdotal event>']
        self.discourse_definiton = "The schema for discourse structure is defined below: "+\
                    "<main event> refers to the major subject of the news article. "+\
                    "<consequence> refers to an event or phenomenon that is caused by the main event. "+\
                    "<previous event> refers to a specific event that occurred shortly before the main event. "+\
                    "<current context> refers to the general context or worldstate immediately preceding the main event. "+\
                    "<historical event> refers to an event occurring much earlier than the main event. "+\
                    "<future consequences> refers to an analytical insight into future consequences or projections made by the journalist. "+\
                    "<journalist evaluation> refers to a summary, opinion or comment made by the journalist. "+\
                    "<anecdotal event> refers to anecdotal events that are uncertain and cannot be verified. The primary purpose is to provide more emotional resonance to the main event. \n\n"

    def __len__(self):
        return len(self.headline)

    def __getitem__(self, idx):    
        if self.prompt_version == 1:
            instruction_prompt = "Polish the following news article about {} to make it more fluent: {}".format(
                    self.headline[idx],
                    self.input_text[idx]
            )

        elif self.prompt_version == 2:
            instruction_prompt = "Polish the following news article about {} to make it more fluent. It should follow the discourse structure of {}: {}".format(
                    self.headline[idx],
                    ' '.join([self.stage_tags[stage] for stage in self.stage_plan[idx]]),
                    self.input_text[idx]
            )

        # self.stage_label[idx]
        return {'instruction': instruction_prompt,
                'target_text': self.target_text[idx],
        }