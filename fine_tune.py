#Fine_Tune.py
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, LineByLineTextDataset

def fine_tune(model, tokenizer, dataset_path):
    # set up the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # set up the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=1000,
        save_total_limit=2
    )
    # create dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128,
    )
    
    # initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    # Fixing up the dataset
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    
    # start the fine-tuning process
    trainer.train()
    return model, tokenizer
