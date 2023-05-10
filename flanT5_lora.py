from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from random import randrange
import argparse


def train(args, model, tokenizer, train_dataset, eval_dataset):

    # huggingface hub model id
    model_id = "philschmid/flan-t5-xxl-sharded-fp16"

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

    # Define LoRA Config
    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    output_dir="lora-flan-t5-xxl"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train() 
    
    # Save our LoRA model & tokenizer results
    peft_model_id="results"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    # if you want to save the base model to call
    # trainer.model.base_model.save_pretrained(peft_model_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='The path to the preprocessed data.')
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
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--gradient_accumulation_step', type=int, default=1)
    parser.add_argument('--input_max_length', type=int, default=1024)
    parser.add_argument('--output_max_length', type=int, default=1024)
    parser.add_argument('--logging_steps', type=int, default=500, help='Print loss every this number of steps.')
    parser.add_argument('--warmup_steps', type=int, default=200)
    args = parser.parse_args()
    # dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))

    if args.dataset not in ['news', 'recipe']:
        raise ValueError('The dataset should be either "news" or "recipe".')

    # Load dataset
    dataset = {}
    dataset['train'] = load_dataset('json', data_files=args.data_path, field='train_data')['train']
    dataset['valid'] = load_dataset('json', data_files=args.data_path, field='valid_data')['train']
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['valid'])}")

    # Load model and tokenizer
    model_id="google/flan-t5-xxl" # 11B parameters
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, 85))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 90))
    print(f"Max target length: {max_target_length}")


    def preprocess_function(sample, padding="max_length"):
        # Get prompt instructions
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # save datasets to disk for later easy loading
    tokenized_dataset["train"].save_to_disk("data/train")
    tokenized_dataset["test"].save_to_disk("data/eval")

if __name__ == '__main__':
    main()  