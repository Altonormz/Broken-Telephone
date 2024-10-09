import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import logging
from dataclasses import dataclass
import contextlib
import os

@dataclass
class TranslationConfig:
    # Model Config
    do_sample: bool = True
    temperature: float = 0.4
    top_p: float = 0.92
    max_length: int = 1000
    num_beams: int = 5
    early_stopping: bool = True

@contextlib.contextmanager
def torch_gc():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class HebrewEnglishTranslator:
    def __init__(self, local_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.local_path = local_path
        self.tokenizer = None
        self.model = None
        self.logger = self._setup_logger()
        self._load_model()
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def _load_model(self):
        try:
            if os.path.exists(self.local_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_path,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                self.logger.info(f"Model loaded successfully from {self.local_path} on {self.device}")
            else:
                raise FileNotFoundError(f"No model found at {self.local_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _create_prompt(self, text: str, english_language: bool) -> str:
        if english_language:
            prompt = f"תתרגם את הטקסט הבא מאגלית לעברית בצורה טבעית ומקצועית\nאנגלית: '{text}'\nעברית:"
        else:
            prompt = f"Translate the following text from Hebrew to English in a natural and professional way\nHebrew: <hebrew_text>'{text}'</hebrew_text>\nEnglish:"
        return prompt

    def translate(self, text: str, english_language: bool=True, config: TranslationConfig = TranslationConfig()) -> Optional[str]:
        try:
            with torch_gc():
                prompt = self._create_prompt(text, english_language)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=config.max_length,
                        num_return_sequences=1,
                        do_sample=config.do_sample,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        early_stopping=config.early_stopping,
                        num_beams=config.num_beams,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id
                        
                    )
                    
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if english_language:
                    translated_text = translation.split("עברית:")[-1].strip()
                else:
                    translated_text = translation.split("English:")[-1].strip()
                translated_text = translated_text.replace("[/INST]", "").strip()
                return translated_text
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return None

