
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.env_vars import LLAMA_MODEL_API

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LlamaInstruct:

    def __init__(self, model_name: str) :
        self.device_name = self.check_gpu()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=LLAMA_MODEL_API,
            add_eos_token=True, pad_token_id=0, padding_side="left"
        )
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_name, trust_remote_code=True, token=LLAMA_MODEL_API)
            .to(self.device_name)
            .eval()
        )

    @staticmethod
    def check_gpu() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def llama_answering(self, system_prompt: str, user_prompt: str) -> str:
        # Create message to be passed to the model
        message = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]

        # Generate input token for the message
        input_text = self.tokenizer.apply_chat_template(message, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device_name)

        # Set the parameters of the model and generate the response
        generator_params = {
            "max_new_tokens": 10000, "temperature": 0.1, "top_p": 0.9, "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id, "eos_token_id": self.tokenizer.eos_token_id
        }
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generator_params)

        return self.tokenizer.decode(outputs[0])
