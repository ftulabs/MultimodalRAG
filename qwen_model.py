#!/usr/bin/env python
"""
Qwen3-VL-4B-Instruct model wrapper
Provides LLM and Vision capabilities using local transformers
"""

import os
import torch
import base64
import logging
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

logger = logging.getLogger(__name__)


class QwenModel:
    """Wrapper class for Qwen3-VL model"""
    
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Instruct"):
        """Initialize Qwen3-VL model and processor"""
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load model and processor"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang tải model {self.model_name} trên {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model {self.model_name} đã được tải thành công!")
    
    def generate_text(
        self, 
        prompt, 
        system_prompt=None, 
        image_data=None, 
        messages=None, 
        max_new_tokens=2048,
        temperature=None,
        top_p=0.9,
        keyword_extraction=False,   
        **kwargs                    
    ):

        """
        Generate text using Qwen3-VL model
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            image_data: Optional base64 encoded image data
            messages: Optional list of conversation messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response (empty string if error)
        """
        try:
            conversation = []

            # Add system prompt if provided
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})

            # Build conversation from messages or prompt
            if messages:
                for msg in messages:
                    if msg and "role" in msg:
                        conversation.append(msg)
            elif image_data:
                # Handle image + text input
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                })
            else:
                # Text-only input
                conversation.append({"role": "user", "content": prompt})

            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepare inputs
            inputs = self.processor(
                text=[text_prompt], 
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            if temperature is None:
                temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=temperature,
                    top_p=top_p,
                )

            # Decode response
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return response

        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    async def llm_func(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """
        LLM function wrapper for RAGAnything
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Conversation history
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        # Remove unsupported arguments
        kwargs.pop("hashing_kv", None)
        
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        
        return self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt if not messages else None,
            messages=messages if messages else None,
            **kwargs
        )
    
    async def vision_func(
        self, 
        prompt, 
        system_prompt=None, 
        history_messages=[], 
        image_data=None, 
        messages=None, 
        **kwargs
    ):
        """
        Vision function wrapper for RAGAnything
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Conversation history
            image_data: Base64 encoded image
            messages: Optional message list
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        kwargs.pop("hashing_kv", None)
        
        return self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            image_data=image_data,
            messages=messages,
            **kwargs
        )


# Singleton instance for global use
_qwen_instance = None

def get_qwen_model(model_name="Qwen/Qwen3-VL-4B-Instruct"):
    """Get or create singleton QwenModel instance"""
    global _qwen_instance
    if _qwen_instance is None:
        _qwen_instance = QwenModel(model_name)
    return _qwen_instance