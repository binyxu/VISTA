import json
import re
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.model_client import ModelClient
from src.method_register import get_method, list_methods
from src.method.base_method import BaseMethod
from utils.extract_final_answer import extract_final_answer
from utils.embedding import EmbeddingEngine
from utils.usage_logger import approx_tokens, usage_context

class MemoryQAInterface:
    """
    Main interface for memory-based QA evaluation.

    Core workflow:
    1. build_memory(trajectory, task) - Build memory from trajectory
    2. answer_question(question, memory) - Answer question using memory
    """

    def __init__(
        self,
        client: ModelClient,
        method_name: str = "longcontext",
        method_config: Optional[str] = None,
        max_concurrency_episodes: int = 1,
        max_concurrency_questions: int = 1,
        subset: str = "mcq",
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize QA interface.

        Args:
            client: Main ModelClient instance for answering questions
            method_name: Memory method to use ("bm25", "embedding", "longcontext")
            method_config: Path to configuration file for the memory method (JSON or YAML)
            max_concurrency_episodes: Max number of episodes to process concurrently
            max_concurrency_questions: Max number of questions per episode to process concurrently
            subset: Dataset subset type ("mcq" or "openend") for answer format
            embedding_engine: Optional embedding engine for semantic embedding
        """
        self.client = client
        self.method_name = method_name
        self.provider = client.provider
        self.model = client.model
        self.max_concurrency_episodes = max_concurrency_episodes
        self.max_concurrency_questions = max_concurrency_questions
        self.subset = subset
        vllm_launch = client.config.get('vllm_launch', {})
        self.max_tokens = (
            client.config.get('max_tokens')
            or vllm_launch.get('max_response_len', 4096)
        )
        self.embedding_engine = embedding_engine

        if method_config:
            self.method = get_method(
                self.method_name,
                config_path=method_config,
                client=client,
                embedding_engine=embedding_engine
            )
        else:
            self.method = get_method(
                self.method_name,
                client=client,
                embedding_engine=embedding_engine
            )


    def _trajectory_to_text(self, trajectory: List[Dict[str, Any]]) -> str:
        """
        Convert trajectory list to text format.

        Args:
            trajectory: List of trajectory steps

        Returns:
            String-formatted trajectory
        """
        text_parts = []

        for step in trajectory:
            turn_idx = step.get("turn_idx", 0)
            action = step.get("action", "")
            observation = step.get("observation", "")
            text_parts.append(f"Step {turn_idx}:")
            text_parts.append(f"Action: {action}")
            text_parts.append(f"Observation: {observation}")
            text_parts.append("")

        return "\n".join(text_parts)

    def memory_construction(
        self,
        trajectory: List[Dict[str, Any]],
        task: str,
        episode_id: Optional[int] = None,
    ) -> Any:
        """
        Build memory from trajectory using the registered method.

        Args:
            trajectory: List of trajectory steps, each containing:
                - turn_idx: Step index
                - action: Action taken
                - observation: Observation received
            task: Task description

        Returns:
            Memory object (format depends on the method)
        """
        # Convert trajectory to text
        traj_text = self._trajectory_to_text(trajectory)

        # Call the method's memory_construction
        with usage_context(stage="construction", episode_id=episode_id, qa_index=None):
            memory = self.method.memory_construction(traj_text, task)

        return memory

    def answer_question(
        self,
        question: str,
        memory: Any,
        temperature: float = 0.0,
        episode_id: Optional[int] = None,
        qa_index: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Answer a question using the memory.

        This method:
        1. Calls memory_retrieve to get relevant information
        2. Uses the retrieved information to answer the question with LLM

        Args:
            question: Question to answer
            memory: Memory object (from memory_construction)
            temperature: Sampling temperature

        Returns:
            Dictionary with 'final_answer' and 'reasoning_trace'
        """
        t0 = time.perf_counter()
        with usage_context(stage="retrieval", episode_id=episode_id, qa_index=qa_index):
            retrieved_context = self.method.memory_retrieve(memory, question)
        retrieval_seconds = time.perf_counter() - t0

        mcq_mode = (self.subset == "mcq")

        # Direct-answer fast path: if the retriever (ama_agent's sufficiency
        # judgment) already produced an answer, skip the second LLM call.
        direct_match = re.match(
            r"<<<AMA_DIRECT_ANSWER>>>(.*?)<<<END_AMA_DIRECT_ANSWER>>>\n",
            retrieved_context,
            re.DOTALL,
        )
        if direct_match:
            direct_answer = extract_final_answer(
                f"###Answer: {direct_match.group(1).strip()}",
                mcq_mode=mcq_mode,
            )
            return {
                'final_answer': direct_answer,
                'reasoning_trace': retrieved_context[direct_match.end():],
                'metrics': {
                    'retrieval_seconds': retrieval_seconds,
                    'answer_seconds': 0.0,
                    'total_seconds': retrieval_seconds,
                    'direct_answer': True,
                    'retrieved_context_tokens_est': approx_tokens(retrieved_context),
                    'answer_prompt_tokens_est': 0,
                    'final_answer_tokens_est': approx_tokens(direct_answer),
                },
            }
        if mcq_mode:
            instructions = (
                "Select all correct options and respond using "
                "the format (A), (B), (C), or (D). "
                "If multiple options are correct, combine them like (A)(B)."
            )
            answer_slot = "Answer[1]: [(A)/(B)/(C)/(D) or combination such as (A)(B)]"
        else:
            instructions = "Provide a direct and concise answer."
            answer_slot = "Answer[1]: [your answer here]"

        prompt = (
            f"{retrieved_context}\n\n"
            f"## Questions\n"
            f"Question 1: {question}\n\n"
            f"## Instructions\n"
            f"{instructions}\n\n"
            f"{answer_slot}"
        )

        answer_t0 = time.perf_counter()
        with usage_context(stage="answer", episode_id=episode_id, qa_index=qa_index):
            response = self.client.query(prompt, temperature=temperature, max_tokens=self.max_tokens)
        answer_seconds = time.perf_counter() - answer_t0

        match = re.search(r"Answer\[1\]:\s*(.+?)$", response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            final_answer = extract_final_answer(f"###Answer: {answer_text}", mcq_mode=mcq_mode)
        else:
            final_answer = extract_final_answer(response, mcq_mode=mcq_mode)

        return {
            'final_answer': final_answer,
            'reasoning_trace': retrieved_context,
            'metrics': {
                'retrieval_seconds': retrieval_seconds,
                'answer_seconds': answer_seconds,
                'total_seconds': retrieval_seconds + answer_seconds,
                'direct_answer': False,
                'retrieved_context_tokens_est': approx_tokens(retrieved_context),
                'answer_prompt_tokens_est': approx_tokens(prompt),
                'raw_response_tokens_est': approx_tokens(response),
                'final_answer_tokens_est': approx_tokens(final_answer),
            },
        }
    

    def _answer_question_with_index(
        self,
        question: str,
        memory: Any,
        qa_index: int,
        episode_id: Optional[int] = None,
    ) -> tuple:
        """Helper method for parallel question answering."""
        result = self.answer_question(question, memory, episode_id=episode_id, qa_index=qa_index)
        return qa_index, result

    def answer_all_questions_batch(self, questions: List[str], memory: Any, temperature: float = 0.0) -> List[str]:
        """
        Answer all questions in a single batch call (for longcontext method).

        Delegates prompt construction (including trajectory truncation) entirely to
        memory_retrieve, which returns a complete create_long_context_prompt-style
        prompt ready to send to the LLM.

        Args:
            questions: List of questions
            memory: Memory object
            temperature: Sampling temperature

        Returns:
            List of final answers
        """
        mcq_mode = (self.subset == "mcq")

        # memory_retrieve builds the complete prompt and handles truncation
        prompt = self.method.memory_retrieve(memory, questions, mcq_mode=mcq_mode)

        # Query LLM once for all questions
        with usage_context(stage="answer_batch"):
            response = self.client.query(prompt, temperature=temperature, max_tokens=self.max_tokens)

        # Parse Answer[{i}]: markers
        answer_list = []
        for i in range(len(questions)):
            pattern = rf"Answer\[{i+1}\]:\s*(.+?)(?=Answer\[{i+2}\]:|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                answer_text = match.group(1).strip()
                final_answer = extract_final_answer(f"###Answer: {answer_text}", mcq_mode=mcq_mode)
            else:
                final_answer = extract_final_answer(response, mcq_mode=mcq_mode)
            answer_list.append(final_answer)

        return answer_list

    def process_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single episode: build memory and answer all questions.

        Args:
            episode_data: Episode data containing task, trajectory, and qa_pairs

        Returns:
            Dictionary with:
            - episode_id: Episode identifier
            - answer_list: List of predicted answers (strings)
            - reasoning_trace: Combined reasoning traces (optional)
        """
        episode_id = episode_data.get("episode_id", 0)
        task = episode_data.get("task", "")
        trajectory = episode_data.get("trajectory", [])
        qa_pairs = episode_data.get("qa_pairs", [])
        episode_start = time.perf_counter()
        construction_start = time.perf_counter()
        memory = self.memory_construction(trajectory, task, episode_id=episode_id)
        construction_seconds = time.perf_counter() - construction_start
        if self.method_name == "longcontext":
            # Batch answering: answer all questions in a single call
            questions = [qa_pair.get("question", "") for qa_pair in qa_pairs]
            answer_list = self.answer_all_questions_batch(questions, memory)
            reasoning_trace = ""
            question_metrics = [
                {
                    'qa_index': i,
                    'question_tokens_est': approx_tokens(q),
                }
                for i, q in enumerate(questions)
            ]
        else:
            # Original parallel answering: one call per question
            answer_list = []
            reasoning_traces = []
            question_metrics = []

            with ThreadPoolExecutor(max_workers=self.max_concurrency_questions) as executor:
                futures = {
                    executor.submit(
                        self._answer_question_with_index,
                        qa_pair.get("question", ""),
                        memory,
                        i,
                        episode_id,
                    ): i
                    for i, qa_pair in enumerate(qa_pairs)
                }

                results_dict = {}
                for future in as_completed(futures):
                    qa_index, result = future.result()
                    results_dict[qa_index] = result

                # Build answer_list and reasoning_traces in order
                for i in range(len(qa_pairs)):
                    result = results_dict[i]
                    answer_list.append(result['final_answer'])
                    reasoning_traces.append(result['reasoning_trace'])
                    metrics = dict(result.get('metrics', {}))
                    metrics.update({
                        'qa_index': i,
                        'question_tokens_est': approx_tokens(qa_pairs[i].get("question", "")),
                    })
                    question_metrics.append(metrics)

            # Combine reasoning traces
            reasoning_trace = "\n\n---\n\n".join([
                f"Q{i+1} Reasoning:\n{trace}"
                for i, trace in enumerate(reasoning_traces)
            ])

        return {
            'episode_id': episode_id,
            'answer_list': answer_list,
            'reasoning_trace': reasoning_trace,
            'metrics': {
                'construction_seconds': construction_seconds,
                'total_episode_seconds': time.perf_counter() - episode_start,
                'trajectory_tokens_dataset': episode_data.get('total_tokens'),
                'trajectory_tokens_est': approx_tokens(self._trajectory_to_text(trajectory)),
                'question_metrics': question_metrics,
            },
        }

    def run(
        self,
        file_path: str,
        episodes: Optional[List[Dict[str, Any]]] = None,
        on_episode_result: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a single JSONL file containing multiple episodes.
        Each episode contains multiple QA pairs.

        Args:
            file_path: Path to JSONL file (e.g., mcq_set.jsonl, open_end_qa_set.jsonl)
            episodes: Optional pre-loaded and pre-filtered list of episodes. If provided,
                      file_path is only used for display purposes and not read again.

        Returns:
            List of episode results, each containing:
            {
                'episode_id': int,
                'answer_list': List[str],  # List of predicted answers
                'reasoning_trace': str     # Combined reasoning traces (optional)
            }
        """
        file_name = Path(file_path).name
        print(f"\n{'='*70}")
        print(f"Processing: {file_name}")
        print(f"{'='*70}")

        # Use pre-loaded episodes if provided, otherwise read from file
        if episodes is None:
            episodes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    episode_data = json.loads(line.strip())
                    episodes.append(episode_data)

        print(f"Total episodes: {len(episodes)}")
        print(f"Max concurrency - Episodes: {self.max_concurrency_episodes}, Questions: {self.max_concurrency_questions}")

        # Process episodes with parallelism
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency_episodes) as executor:
            futures = {
                executor.submit(self.process_episode, episode): episode.get('episode_id', idx)
                for idx, episode in enumerate(episodes)
            }

            # Use tqdm for progress bar
            with tqdm(total=len(episodes), desc="Processing episodes", unit="episode") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    if on_episode_result is not None:
                        on_episode_result(result)
                    pbar.update(1)
                    pbar.set_postfix({"Episode": result['episode_id'], "Questions": len(result['answer_list'])})

        # Sort results by episode_id to maintain order
        all_results.sort(key=lambda x: x['episode_id'])

        return all_results
