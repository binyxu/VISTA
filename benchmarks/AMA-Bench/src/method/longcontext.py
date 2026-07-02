"""
Long Context Method - Uses the full trajectory as context without compression or retrieval.

This method does not call any embedding service and is suitable for offline or
network-restricted runs.
"""

from typing import Any, List, Optional, Tuple, Union
import yaml
from pathlib import Path

from src.method.base_method import BaseMethod


class LongContextMemory:
    """Memory object for long context method - simply stores the full text"""

    def __init__(self, full_text: str):
        self.full_text = full_text


class LongContextMethod(BaseMethod):
    """
    Long context method.

    Uses the entire trajectory as context without any compression or retrieval.
    Suitable for models with large context windows.
    Does not require embeddings or network access for memory construction/retrieval.
    """

    def __init__(self, config_path: str = None, embedding_engine: Any = None):
        """
        Initialize long context method.

        Args:
            config_path: Path to configuration file (optional)
                        Config can specify: max_model_length, max_response_tokens
            embedding_engine: Optional embedding engine (ignored by LongContext;
                              kept only for API compatibility)
        """
        # Default values
        self.max_model_length = 16384  # Model's maximum context length
        self.max_response_tokens = 4096  # Reserved for response
        self.safety_buffer = 300       # Fixed overhead for prompt template/formatting tokens
        self.tokenizer = None
        self.requires_embedding = False
        self.requires_network = False

        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            vllm_launch = config.get('vllm_launch', {})
            self.max_model_length = (
                config.get('max_model_length')
                or vllm_launch.get('max_model_len', self.max_model_length)
            )
            # Align with the actual max_tokens the client will request
            self.max_response_tokens = (
                config.get('max_response_tokens')
                or vllm_launch.get('max_response_len', self.max_response_tokens)
            )

            # Try to load the model's tokenizer for accurate token counting
            model_path = config.get('model')
            if model_path:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception:
                    self.tokenizer = None

        self.embedding_engine = embedding_engine  # Not used, for compatibility

    def _load_config(self, config_path: str) -> dict:
        with open(Path(config_path), 'r') as f:
            return yaml.safe_load(f)

    def _encode_prompt_tokens(self, prompt: str):
        """Return token id list for prompt, or None if tokenizer unavailable."""
        if self.tokenizer is None:
            return None
        try:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        except Exception:
            return None

    def truncate_prompt(self, prompt: str, target_length: Optional[int] = None) -> Tuple[str, Optional[int]]:
        """Truncate prompt to fit within the context window.

        Keeps the first 70% and last 30% of the allowed budget so that
        recent context (end of trajectory) is always preserved.

        Returns:
            (truncated_prompt, actual_token_count_or_None)
        """
        if target_length is not None:
            max_allowed = target_length
        elif self.max_model_length is not None:
            max_allowed = self.max_model_length - self.max_response_tokens - self.safety_buffer
        else:
            return prompt, None

        max_allowed = max(100, max_allowed)

        token_ids = self._encode_prompt_tokens(prompt)
        if token_ids is None:
            return prompt, None

        token_count = len(token_ids)
        if token_count <= max_allowed:
            return prompt, token_count

        head_tokens = int(max_allowed * 0.7)
        tail_tokens = max_allowed - head_tokens

        truncated_ids = token_ids[:head_tokens]
        if tail_tokens > 0:
            truncated_ids = truncated_ids + token_ids[-tail_tokens:]

        try:
            truncated_prompt = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        except Exception:
            truncated_prompt = self.tokenizer.decode(truncated_ids)

        truncated_len = len(self._encode_prompt_tokens(truncated_prompt) or []) or None
        #print(f"  Prompt tokens {token_count} exceed limit {max_allowed}; truncated to {truncated_len} tokens.")
        return truncated_prompt, truncated_len

    def memory_construction(self, traj_text: str, task: str = "") -> LongContextMemory:
        full_text = traj_text
        if task:
            full_text = (
                f"## Task Description\n{task}\n\n"
                f"## Agent Trajectory\n"
                f"The following is a step-by-step trajectory of the agent's actions and observations:\n\n"
                f"{traj_text}"
            )
        return LongContextMemory(full_text)

    def memory_retrieve(
        self,
        memory: LongContextMemory,
        questions: Union[str, List[str]],
        mcq_mode: bool = False,
    ) -> str:
        """Retrieve context from memory, accounting for question overhead.

        When *questions* is a plain string (single-question / backward-compat mode)
        the method returns the truncated context only, and the caller is responsible
        for assembling the final prompt.

        When *questions* is a list the method builds and returns the **complete**
        prompt in ``create_long_context_prompt`` format: context → questions →
        instructions → answer slots.  The trajectory section is truncated so that
        the whole prompt fits within the model's context window.

        Args:
            memory: LongContextMemory object produced by memory_construction.
            questions: Either a single question string or a list of question strings.
            mcq_mode: Whether to use MCQ answer format (default False / open-ended).

        Returns:
            Truncated context string (str input) or complete formatted prompt (list input).
        """
        if not isinstance(memory, LongContextMemory):
            raise ValueError("Memory must be a LongContextMemory object")

        if isinstance(questions, str):
            # Single question: return truncated context only (caller builds the prompt)
            question_ids = self._encode_prompt_tokens(questions)
            question_overhead = len(question_ids) if question_ids else 0
            target = (
                self.max_model_length
                - self.max_response_tokens
                - self.safety_buffer
                - question_overhead
            )
            truncated, _ = self.truncate_prompt(memory.full_text, target_length=max(100, target))
            return truncated

        # --- List mode: build the complete create_long_context_prompt-style prompt ---
        questions_block = "\n".join(
            f"Question {i}: {q}\n"
            for i, q in enumerate(questions, 1)
        )

        if mcq_mode:
            section_intro = (
                "Please answer the following multiple-choice questions based on "
                "the task description and agent trajectory above."
            )
            instructions = (
                "For each question, select all correct options and respond using "
                "the format (A), (B), (C), or (D). "
                "If multiple options are correct, combine them like (A)(B)."
            )
            answer_slots = "\n".join(
                f"Answer[{i}]: [(A)/(B)/(C)/(D) or combination such as (A)(B)]"
                for i in range(1, len(questions) + 1)
            )
        else:
            section_intro = (
                "Please answer the following questions based on the task description "
                "and agent trajectory above. For each question, provide a direct and "
                "concise answer."
            )
            instructions = "Please provide answers in the following format:"
            answer_slots = "\n".join(
                f"Answer[{i}]: [your answer here]"
                for i in range(1, len(questions) + 1)
            )

        suffix = (
            f"\n\n## Questions\n{section_intro}\n\n"
            f"{questions_block}\n"
            f"## Instructions\n{instructions}\n\n"
            f"{answer_slots}"
        )

        # Deduct suffix tokens from available budget so the full prompt fits
        suffix_ids = self._encode_prompt_tokens(suffix)
        suffix_overhead = len(suffix_ids) if suffix_ids else 0
        target = (
            self.max_model_length
            - self.max_response_tokens
            - self.safety_buffer
            - suffix_overhead
        )
        truncated_context, _ = self.truncate_prompt(memory.full_text, target_length=max(100, target))
        return truncated_context + suffix
