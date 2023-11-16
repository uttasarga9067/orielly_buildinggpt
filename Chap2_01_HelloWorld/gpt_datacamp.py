import torch
import textwrap
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(train_dataset)


pandas_format = train_dataset.to_pandas()
# display(pandas_format.head())

for index in range(3):
   print("---"*15)



pretrained_model_name = "Salesforce/xgen-7b-8k-base"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)


model_training_args = TrainingArguments(
       output_dir="./xgen-7b-8k-base-fine-tuned",
       per_device_train_batch_size=1,
       optim="adamw_torch",
       logging_steps=80,
       learning_rate=2e-4,
       warmup_ratio=0.1,
       lr_scheduler_type="linear",
       num_train_epochs=1,
       save_strategy="epoch"
   )



SFT_trainer = SFTTrainer(
       model=model,
       train_dataset=train_dataset,
       dataset_text_field="text",
       max_seq_length=1024,
       tokenizer=tokenizer,
       args=model_training_args,
       packing=True,
    #    peft_config=lora_peft_config,
   )



tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_peft_config)
training_args = model_training_args
trainer = SFT_trainer
trainer.train()
