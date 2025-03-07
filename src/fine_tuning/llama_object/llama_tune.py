
import torch
import torch.nn as nn
import random
import numpy as np
import os
import shutil
from datasets import Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.fine_tuning.llama_object import llama_instruct

DATA_LOCATION = Path(__file__).joinpath("..", "..", "..", "..", "data_dump").resolve()


class LlamaTune(llama_instruct.LlamaInstruct):

    def __init__(self, model_name: str, r: int = 16, alpha: int = 16, dropout: float = 0.05):
        super().__init__(model_name)
        self._freeze_params()
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.peft_model = self._lora_configurations(self.model, self.r, self.alpha, self.dropout)
        self.fp16_param = True if self.device_name == "cuda" else False

    @staticmethod
    def _set_device_seed(seed: int) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)

        elif torch.backends.mps.is_available():
            torch.backends.mps.deterministic = True
            torch.backends.mps.benchmark = False
            torch.mps.manual_seed(seed)

        else:
            torch.backends.cpu.deterministic = True
            torch.backends.cpu.benchmark = False

    def _freeze_params(self) -> None:
        # Freeze the original model parameters, only train on new LoRA weights
        for param in self.model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        # Ensure that the final output logits of the model are full precision
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        self.model.gradient_checkpointing_enable()
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)

    @staticmethod
    def _lora_configurations(model, r: int, alpha: int, dropout: float) -> PeftModel:
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        config.inference_mode = False

        return get_peft_model(model, config)

    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = 0, 0

        # Loop through the model and count the trainable parameters
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}")

    def model_fine_tuning(
        self, seed: int, training_data: Dataset, folder_name: str, model_name: str,
        warmup_steps: int, max_steps: int
    ) -> None:
        # Create folder if not exists
        output_folder = DATA_LOCATION.joinpath(folder_name).resolve()
        os.makedirs(output_folder, exist_ok=True)

        # Set seed to make sure reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._set_device_seed(seed)

        # Create the trainer
        trainer = Trainer(
            model=self.peft_model, train_dataset=training_data,
            args=TrainingArguments(
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                warmup_steps=warmup_steps, max_steps=max_steps, learning_rate=1e-3, fp16=self.fp16_param,
                logging_steps=1, output_dir="outputs", weight_decay=0.01
            ),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        self.peft_model.config.use_cache = False

        # Run the training model and save the final model
        model_path = output_folder.joinpath(model_name).resolve()
        shutil.rmtree(model_path, ignore_errors=True)
        trainer.train()
        self.peft_model.save_pretrained(model_path.__str__())
        self.tokenizer.save_pretrained(model_path.__str__())
