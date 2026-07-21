import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class SLMGenerator:
    def __init__(self):
        self.model_name = "DeepHat/DeepHat-V1-7B"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
        )

        self.model.eval()

    def generate(self, query: str, context: str) -> str:
        if not context or not context.strip():
            return "I don't have enough information in the retrieved documents."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity assistant. "
                    "Use only the retrieved context to answer. "
                    "Do not guess or invent information. "
                    "Provide a clear and sufficiently detailed explanation."
                    "Do not continue generating after the answer is complete."
                    "End the response immediately after finishing the explanation."
                ),
            },
            {
                "role": "user",
                "content": f"""
Retrieved context:
{context}

Question:
{query}

Answer the question using only the retrieved context. Combine relevant
information into a coherent explanation. If the context is insufficient,
reply exactly: "I don't have enough information in the retrieved documents."
""",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        input_device = next(self.model.parameters()).device
        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()
