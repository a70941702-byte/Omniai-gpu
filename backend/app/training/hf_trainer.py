"""Optional real Transformers/PEFT training pipeline. Imports are lazy so CPU-only installs keep working."""
from __future__ import annotations
from pathlib import Path

class HFTrainingPipeline:
    @staticmethod
    def available():
        try:
            import transformers, peft
            return True
        except ImportError: return False

    def train_lora(self, model_name, dataset, output_dir, epochs=1, lr=2e-4, qlora=False,
                   gradient_accumulation_steps=4, resume_from_checkpoint=None):
        try:
            from datasets import Dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e: raise RuntimeError("transformers,datasets,peft are required") from e
        import torch
        tokenizer=AutoTokenizer.from_pretrained(model_name); tokenizer.pad_token=tokenizer.eos_token
        kwargs={}
        if qlora:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
                kwargs["device_map"]="auto"
            except ImportError as e: raise RuntimeError("bitsandbytes/transformers 4-bit support required for QLoRA") from e
        model=AutoModelForCausalLM.from_pretrained(model_name,**kwargs)
        if qlora: model=prepare_model_for_kbit_training(model)
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",task_type="CAUSAL_LM"))
        ds=Dataset.from_list(dataset)
        def tok(x): return tokenizer(x["text"],truncation=True,max_length=tokenizer.model_max_length)
        ds=ds.map(tok,batched=True,remove_columns=ds.column_names)
        args=TrainingArguments(output_dir=str(Path(output_dir)),num_train_epochs=epochs,learning_rate=lr,
            per_device_train_batch_size=1,gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),bf16=False,save_strategy="steps",save_steps=100,
            logging_steps=10,report_to=[])
        trainer=Trainer(model=model,args=args,train_dataset=ds,data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False))
        result=trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
        return {"output_dir":str(output_dir),"global_step":result.global_step,"train_loss":result.training_loss}

hf_training=HFTrainingPipeline()
