import asyncio
import dataclasses
from typing import Any

import torch


@dataclasses.dataclass
class Request:
    prompt: str
    future: asyncio.Future
    input_ids: torch.Tensor
    output_tokens: list[int] = dataclasses.field(default_factory=list)
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    finished: bool = False

class ContinuousBatchingEngine:
    def __init__(self, inference_engine: Any, max_batch_size: int = 8):
        """
        Initialize continuous batching engine.
        
        Args:
            inference_engine: Instance of MetalInferenceEngine or compatible
            max_batch_size: Maximum number of concurrent requests
        """
        self.engine = inference_engine
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue[Request] = asyncio.Queue()
        self.active_requests: list[Request] = []
        self.running = False
        self._loop_task: asyncio.Task | None = None

    async def start(self):
        """Start the processing loop."""
        self.running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the processing loop."""
        self.running = False
        if self._loop_task:
            await self._loop_task

    async def add_request(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Add a request to the queue.
        
        Returns:
            Generated text
        """
        if not self.running:
            raise RuntimeError("Engine is not running. Call start() first.")

        # Encode prompt
        input_ids = self.engine.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        input_ids = input_ids.to(self.engine.device)

        future = asyncio.Future()
        req = Request(
            prompt=prompt,
            future=future,
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        await self.queue.put(req)
        return await future

    async def _run_loop(self):
        while self.running:
            # 1. Fill batch from queue
            while len(self.active_requests) < self.max_batch_size and not self.queue.empty():
                try:
                    req = self.queue.get_nowait()
                    self.active_requests.append(req)
                except asyncio.QueueEmpty:
                    break

            if not self.active_requests:
                await asyncio.sleep(0.001)
                continue

            # 2. Process batch
            # We wrap step in a try-except block to ensure failures propagate to futures
            try:
                self._step()
            except Exception as e:
                for req in self.active_requests:
                    if not req.future.done():
                        req.future.set_exception(e)
                self.active_requests = []
                continue

            # 3. Handle finished requests
            remaining = []
            for req in self.active_requests:
                if req.finished:
                    if not req.future.done():
                        # Construct full text
                        full_ids = torch.cat([
                            req.input_ids,
                            torch.tensor(req.output_tokens, device=req.input_ids.device)
                        ])
                        text = self.engine.tokenizer.decode(full_ids)
                        req.future.set_result(text)
                else:
                    remaining.append(req)

            self.active_requests = remaining

            # Small yield to prevent locking up the loop completely
            await asyncio.sleep(0.0)

    def _step(self):
        if not self.active_requests:
            return

        # Prepare batch tensors
        current_seqs = []
        for req in self.active_requests:
            if req.output_tokens:
                generated = torch.tensor(req.output_tokens, device=req.input_ids.device)
                full = torch.cat([req.input_ids, generated])
            else:
                full = req.input_ids
            current_seqs.append(full)

        # Padding
        max_len = max(s.size(0) for s in current_seqs)
        batch_size = len(current_seqs)

        pad_token_id = self.engine.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.engine.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0

        padded_input = torch.full(
            (batch_size, max_len),
            pad_token_id,
            device=self.engine.device,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            device=self.engine.device,
            dtype=torch.long
        )

        # Right padding (sequences align to left)
        # Assuming model handles position_ids/masking correctly for this
        for i, seq in enumerate(current_seqs):
            l = seq.size(0)
            padded_input[i, :l] = seq
            attention_mask[i, :l] = 1

        # Forward pass
        # Note: We assume the model accepts attention_mask.
        # Most HF-compatible models do.
        logits = self.engine.model(padded_input, attention_mask=attention_mask)
        # logits: [batch, seq, vocab]

        # Sample next token for each request
        for i, req in enumerate(self.active_requests):
            seq_len = current_seqs[i].size(0)
            # Logits for the last real token
            next_token_logits = logits[i, seq_len - 1, :]

            # Sample
            next_token = self.engine._sample(
                next_token_logits,
                req.temperature,
                req.top_p
            )

            token_id = next_token.item()
            req.output_tokens.append(token_id)

            # Check stopping criteria
            if token_id == self.engine.tokenizer.eos_token_id:
                req.finished = True
            elif len(req.output_tokens) >= req.max_tokens:
                req.finished = True
