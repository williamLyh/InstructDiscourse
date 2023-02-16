from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
import json
import torch 
from tqdm import tqdm
import random
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader

class T5StepLevelTrainingDataset(Dataset):
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

def batch_encode(input, reference, tokenizer, input_max_length=1024, output_max_length=1024):
    encoded_input = tokenizer(input, 
                    max_length=input_max_length, 
                    truncation=True,
                    padding=True,
                    return_tensors='pt')

    encoded_output = tokenizer(reference,
                max_length=output_max_length,
                truncation=True,
                padding=True,
                return_tensors='pt')

    encoding = {'input_ids': encoded_input.input_ids.squeeze(0)}
    encoding['labels'] = encoded_output.input_ids.squeeze(0)
    encoding['attention_mask'] = encoded_input.attention_mask.squeeze(0)
    encoding['labels'][encoding['labels'] == tokenizer.pad_token_id] = -100
    return encoding

class T5StepLevelInferenceDataset(Dataset):
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


def eval_model(args, model, data_loader, device):
    loss = 0
    eval_step = int(len(data_loader)*0.1)
    pbar = tqdm(total=eval_step)
    model.eval()
    with torch.no_grad():
        for idx in range(eval_step):
            pbar.update(1)
            batch = next(iter(data_loader))
            batch_encoding = batch_encode(batch['instruction'], 
                        batch['target_text'], 
                        tokenizer, 
                        input_max_length=1024, 
                        output_max_length=256
                        )

            if args.cuda_available:
                batch_input_ids = batch_encoding['input_ids'].cuda(device)
                batch_attention_mask = batch_encoding['attention_mask'].cuda(device)
                batch_labels = batch_encoding['labels'].cuda(device)

            outputs = model(input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask, 
                            labels=batch_labels)
            # print(outputs.loss)
            # assert False
            loss += outputs.loss.mean().item()            

            # mle_loss, cl_loss = model(batch_input_ids, batch_attention_mask, args.margin)
            # mle_loss_total += mle_loss.mean().item()
            # cl_loss_total += cl_loss.mean().item()

        print('Evaluate on validation set. The loss is {}'.format(loss/(eval_step*args.batch_size)))
    pbar.close()
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_path', help='The path to the preprocessed data.')
    parser.add_argument('--model_saving_path', help='The path to save the trained model.')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2_decay', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--logging_steps', type=int, default=500, help='Print loss every this number of steps.')
    parser.add_argument('--warmup_steps', type=int, default=200)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args.cuda_available = cuda_available
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
    device = torch.device('cuda')


    # kaggle_data_path = '/home/yinhong/Documents/datasets/Kaggle_all_the_news/preprocessed_data.json'
    with open(args.preprocessed_data_path) as outfile:
        data = json.load(outfile)

    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    train_dataset = T5StepLevelTrainingDataset(data['train_data'],
                                            tokenizer=tokenizer)

    valid_dataset = T5StepLevelTrainingDataset(data['valid_data'],
                                            tokenizer=tokenizer)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = args.epoch * len(train_data_loader)
    print('total training steps is {}.\n Warmup steps is {}.\n Loss print for every {} step.\n'.format(total_steps, 
        args.warmup_steps, args.logging_steps))
    print('Epoch number is {}.\n Batch size is {}.'.format(args.epoch, args.batch_size))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    
    model.train()

    global_step = 0
    pbar=tqdm(total=total_steps)
    loss_sum = 0
    for epoch in range(args.epoch):
        for batch in train_data_loader:
            global_step += 1
            pbar.update(1)
            batch_encoding = batch_encode(batch['instruction'], 
                        batch['target_text'], 
                        tokenizer, 
                        input_max_length=1024, 
                        output_max_length=256
                        )
            if cuda_available:
                batch_input_ids = batch_encoding['input_ids'].cuda(device)
                batch_attention_mask = batch_encoding['attention_mask'].cuda(device)
                batch_labels = batch_encoding['labels'].cuda(device)

            outputs = model(input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask, 
                            labels=batch_labels)

            loss = outputs.loss.mean()
            loss_sum += loss.item()
            loss.backward()
            
            # parameter update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # print intermediate result
            if global_step % args.logging_steps == 0:
                average_denominator = global_step * args.batch_size
                print ('At training steps {}/{}, training loss is {},'.format(global_step, total_steps, loss_sum/args.logging_steps))
                print('The learning rate is {}.'.format(scheduler.get_last_lr()[0]))
                loss_sum = 0

            # intermediate evaluation using validation data 
            if global_step % args.eval_steps == 0:
                print('Start evaluating the model on validation dataset: ')
                eval_model(args, model, valid_data_loader, device)

            if global_step % args.save_steps == 0:
                # save model
                full_ckpt_save_path = args.model_saving_path + 'checkpoint-{}'.format(global_step)
                print('Saving model at ' + full_ckpt_save_path)

                if os.path.exists(full_ckpt_save_path):
                    pass
                else: 
                    os.makedirs(full_ckpt_save_path, exist_ok=True)
                # save model
                if multi_gpu_training:
                    model.module.save_pretrained(full_ckpt_save_path)
                    tokenizer.save_pretrained(full_ckpt_save_path)
                else:
                    model.save_pretrained(full_ckpt_save_path)
                    tokenizer.save_pretrained(full_ckpt_save_path)

    pbar.close()