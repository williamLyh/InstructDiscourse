from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.optimization import get_cosine_schedule_with_warmup
import os
import json
import torch 
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from dataclass import T5StepLevelNewsTrainingDataset, T5StepLevelNewsInferenceDataset, T5StepLevelRecipeTrainingDataset
import torch.distributed as dist
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP


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


def eval_model(args, model, tokenizer, data_loader, device=None):
    loss = 0
    eval_step = int(len(data_loader)*0.1)
    print('Evaluate on validation set. The number of steps is {}'.format(eval_step))
    pbar = tqdm(total=eval_step)
    model.eval()
    with torch.no_grad():
        for idx in range(eval_step):
            pbar.update(1)
            batch = next(iter(data_loader))
            batch_encoding = batch_encode(batch['instruction'], 
                        batch['target_text'], 
                        tokenizer, 
                        input_max_length=args.input_max_length, 
                        output_max_length=args.output_max_length
                        )

            if args.cuda_available:
                batch_input_ids = batch_encoding['input_ids']#.cuda(device)
                batch_attention_mask = batch_encoding['attention_mask']#.cuda(device)
                batch_labels = batch_encoding['labels']#.cuda(device)

            outputs = model(input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask, 
                            labels=batch_labels)

            loss += outputs.loss.mean().item()            

        print('Evaluate on validation set. The loss is {}'.format(loss/(eval_step*args.batch_size)))
    pbar.close()
    model.train()

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def cleanup():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_path', help='The path to the preprocessed data.')
    parser.add_argument('--model_saving_path', help='The path to save the trained model.')
    # When the model is continued to be trained, this should be set to the name of the model.
    parser.add_argument('--model_name', help='The name of the model to be trained.', default=None)
    # When the model is continued to be trained, this should be set to the number of steps the model has been trained.
    parser.add_argument('--beginning_step', type=int, help='The step to begin training.', default=0) 
    parser.add_argument('--dataset', type=str, help='Which dataset to train on. News or Recipes', default='news')
    parser.add_argument('--prompt_version', type=int, help='Which prompt instruction to use.', default=1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2_decay', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size_per_gpu', type=int)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--gradient_accumulation_step', type=int, default=1)
    parser.add_argument('--input_max_length', type=int, default=1024)
    parser.add_argument('--output_max_length', type=int, default=1024)
    parser.add_argument('--logging_steps', type=int, default=500, help='Print loss every this number of steps.')
    parser.add_argument('--warmup_steps', type=int, default=200)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    # dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))

    if args.dataset not in ['news', 'recipe']:
        raise ValueError('The dataset should be either "news" or "recipe".')
    
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    # device = torch.device('cuda')

    if torch.cuda.is_available():
        accelerator.print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    gpu_count = 0
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            gpu_count = torch.cuda.device_count()
            accelerator.print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            gpu_count = 1
            accelerator.print ('Using single GPU training.')
    else:
        pass

    args.cuda_available = cuda_available
    args.multi_gpu_training = multi_gpu_training
    args.batch_size = args.batch_size_per_gpu #* gpu_count

    with open(args.preprocessed_data_path) as outfile:
        data = json.load(outfile)

    if args.model_name:
        # continue training
        model_name = args.model_name
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        # Add special tokens, '<' is not in the vocab of Flan-T5
        special_tokens_dict = {'additional_special_tokens': ['<']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    # local_rank = int(os.environ['LOCAL_RANK'])
    # model = DDP(model, device_ids=[local_rank])
    # if args.multi_gpu_training:
    #     model = torch.nn.parallel.DataParallel(model)
 
    if args.dataset == 'news':
        train_dataset = T5StepLevelNewsTrainingDataset(data['train_data'],
                                                tokenizer=tokenizer,
                                                prompt_version=args.prompt_version)

        valid_dataset = T5StepLevelNewsTrainingDataset(data['valid_data'],
                                                tokenizer=tokenizer,
                                                prompt_version=args.prompt_version)
    elif args.dataset == 'recipe':
        train_dataset = T5StepLevelRecipeTrainingDataset(data['train_data'],
                                                tokenizer=tokenizer,
                                                prompt_version=args.prompt_version)

        valid_dataset = T5StepLevelRecipeTrainingDataset(data['valid_data'],
                                                tokenizer=tokenizer,
                                                prompt_version=args.prompt_version)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size*5, shuffle=True)

    # print(train_dataset[0]['instruction'])
    # print(train_dataset[1]['instruction'])
    # assert False

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps, 
                                                num_training_steps=args.epoch * len(train_data_loader)//args.gradient_accumulation_step
                                                )
    optimizer.zero_grad()


    model, optimizer, train_data_loader, valid_data_loader, scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, valid_data_loader, scheduler
    )

    # The size of train_data_loader is divided by number of GPU.
    total_steps = args.epoch * len(train_data_loader)
    if args.beginning_step != 0:
        total_steps += args.beginning_step
    accelerator.print('total training steps is {}, continue from {}.\n Warmup steps is {}.\n Loss print for every {} step.\n'.format(total_steps, 
        args.beginning_step, args.warmup_steps, args.logging_steps))
    accelerator.print('Epoch number is {}.\n Batch size is {}.'.format(args.epoch, args.batch_size))

    accelerator.print ('--------------------------------------------------------------------------')
    accelerator.print ('Start Training:')

    model.train()

    global_step = args.beginning_step
    pbar=tqdm(total=total_steps, initial=args.beginning_step)
    loss_sum_log = 0
    loss_sum_to_optimize = 0
    for epoch in range(args.epoch):
        for idx, batch in enumerate(train_data_loader):
            global_step += 1
            pbar.update(1)
            batch_encoding = batch_encode(batch['instruction'], 
                        batch['target_text'], 
                        tokenizer, 
                        input_max_length=args.input_max_length, 
                        output_max_length=args.output_max_length
                        )
            if cuda_available:
                batch_input_ids = batch_encoding['input_ids'].to(device)
                batch_attention_mask = batch_encoding['attention_mask'].to(device)
                batch_labels = batch_encoding['labels'].to(device)
                # batch_input_ids = batch_encoding['input_ids']
                # batch_attention_mask = batch_encoding['attention_mask']
                # batch_labels = batch_encoding['labels']

            outputs = model(input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask, 
                            labels=batch_labels)
            # print(outputs)
            # assert False
            loss = outputs.loss.mean() / args.gradient_accumulation_step
            loss_sum_log += outputs.loss.mean().item()
            # loss.backward()                
            accelerator.backward(loss)

            if global_step % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # print intermediate result
            if global_step % args.logging_steps == 0:
                average_denominator = global_step * args.batch_size
                accelerator.print ('At training steps {}/{}, training loss is {},'.format(global_step, total_steps, loss_sum_log/args.logging_steps))
                accelerator.print('The learning rate is {}.'.format(scheduler.get_last_lr()[0]))
                loss_sum_log = 0

            # intermediate evaluation using validation data 
            if global_step % args.eval_steps == 0:
                accelerator.print('Start evaluating the model on validation dataset: ')
                # eval_model(args, model, tokenizer, valid_data_loader, device)
                eval_model(args, model, tokenizer, valid_data_loader)

            if global_step % args.save_steps == 0:
                # save model
                full_ckpt_save_path = args.model_saving_path + 'checkpoint-{}'.format(global_step)
                accelerator.print('Saving model at ' + full_ckpt_save_path)

                if os.path.exists(full_ckpt_save_path):
                    pass
                else: 
                    os.makedirs(full_ckpt_save_path, exist_ok=True)
                # save model
                if args.multi_gpu_training:
                    try:
                        model.module.save_pretrained(full_ckpt_save_path)
                    except:
                        model.save_pretrained(full_ckpt_save_path)
                    tokenizer.save_pretrained(full_ckpt_save_path)
                else:
                    model.save_pretrained(full_ckpt_save_path)
                    tokenizer.save_pretrained(full_ckpt_save_path)

    pbar.close()

if __name__ == '__main__':
    main()