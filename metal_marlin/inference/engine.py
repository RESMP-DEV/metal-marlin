'''Unified inference API for MMFP4.

Automatically selects optimal decode strategy:
1. Speculative (MTP) - if MTP head loaded
2. Standard - default fallback
'''

import torch
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    '''Configuration for text generation.'''
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Speculative decoding options
    use_speculative: bool = True  # Auto-enable if MTP available
    num_draft_tokens: int = 4


@dataclass
class GenerationOutput:
    '''Output from generation.'''
    sequences: torch.Tensor  # Generated token IDs [B, S]
    
    # Optional metadata
    acceptance_rate: Optional[float] = None
    tokens_per_second: Optional[float] = None
    decode_strategy: str = "standard"


class InferenceEngine:
    '''High-level inference engine for MMFP4 models.'''
    
    def __init__(
        self,
        model,
        tokenizer=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._has_mtp = hasattr(model, 'mtp_head') and model.mtp_head is not None
    
    @property
    def can_speculate(self) -> bool:
        '''Whether speculative decoding is available.'''
        return self._has_mtp
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        '''Generate text with automatic strategy selection.
        
        Args:
            input_ids: Input token IDs [B, S]
            config: Generation configuration
            
        Returns:
            GenerationOutput with generated sequences and metadata
        '''
        import time
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.perf_counter()
        
        # Select decode strategy
        use_spec = config.use_speculative and self.can_speculate
        
        if use_spec:
            # Speculative decoding with MTP
            sequences, stats = self.model.generate_speculative(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                num_draft_tokens=config.num_draft_tokens,
                eos_token_id=config.eos_token_id,
            )
            
            elapsed = time.perf_counter() - start_time
            new_tokens = sequences.shape[-1] - input_ids.shape[-1]
            
            return GenerationOutput(
                sequences=sequences,
                acceptance_rate=stats.get("acceptance_rate"),
                tokens_per_second=new_tokens / elapsed if elapsed > 0 else 0,
                decode_strategy="speculative",
            )
        else:
            # Standard greedy/sampling decode
            sequences = self._standard_generate(input_ids, config)
            
            elapsed = time.perf_counter() - start_time
            new_tokens = sequences.shape[-1] - input_ids.shape[-1]
            
            return GenerationOutput(
                sequences=sequences,
                tokens_per_second=new_tokens / elapsed if elapsed > 0 else 0,
                decode_strategy="standard",
            )
    
    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        '''Standard autoregressive generation.'''
        generated = input_ids.clone()
        
        for _ in range(config.max_new_tokens):
            outputs = self.model(generated)
            next_logits = outputs.logits[:, -1, :]
            
            if config.do_sample:
                # Temperature + sampling
                next_logits = next_logits / config.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if (next_token == config.eos_token_id).all():
                break
        
        return generated
    
    def chat(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        '''Chat-style interface.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            config: Generation configuration
            
        Returns:
            Assistant response text
        '''
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required for chat interface")
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        
        output = self.generate(input_ids, config)
        
        # Decode only new tokens
        new_tokens = output.sequences[0, input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response
