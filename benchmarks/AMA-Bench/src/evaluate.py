import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.model_client import ModelClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.evaluation_metrics import compute_f1_score, compute_llm_as_judge
from utils.usage_logger import approx_tokens, usage_context


def evaluate_batch(
    qa_results: List[Dict[str, Any]],
    judge_client: ModelClient,
    max_workers: int = 3,
    continue_on_error: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of QA results using LLM-as-judge with concurrent execution.

    Args:
        qa_results: List of QA result dictionaries containing:
            - episode_id, question, predicted_answer, golden_answer, etc.
        judge_client: ModelClient for LLM judge
        max_workers: Maximum number of concurrent workers

    Returns:
        List of evaluated results with scores added
    """
    

    def evaluate_single(result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single QA result."""
        t0 = time.perf_counter()
        with usage_context(
            stage="judge",
            episode_id=result.get('episode_id'),
            qa_index=result.get('qa_index'),
        ):
            score = compute_llm_as_judge(
                question=result['question'],
                golden_answer=result['golden_answer'],
                predicted_answer=result['predicted_answer'],
                judge_client=judge_client,
                task_description=result.get('task_description', ''),
                task_type=result.get('task_type', ''),
                episode_id=str(result.get('episode_id', '')),
            )
        result['score'] = score
        result['f1'] = compute_f1_score(
            result['predicted_answer'],
            result['golden_answer'],
        )
        result['judge_seconds'] = time.perf_counter() - t0
        result['question_tokens_est'] = approx_tokens(result['question'])
        result['golden_answer_tokens_est'] = approx_tokens(result['golden_answer'])
        result['predicted_answer_tokens_est'] = approx_tokens(result['predicted_answer'])
        return result

    # Use thread pool for concurrent evaluation
    evaluated_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_result = {
            executor.submit(evaluate_single, result): result
            for result in qa_results
        }

        with tqdm(total=len(qa_results), desc="Evaluating QA pairs", unit="pair") as pbar:
            for future in as_completed(future_to_result):
                original = future_to_result[future]
                try:
                    evaluated_results.append(future.result())
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    failed = dict(original)
                    failed['score'] = None
                    failed['f1'] = compute_f1_score(
                        failed.get('predicted_answer', ''),
                        failed.get('golden_answer', ''),
                    )
                    failed['judge_error'] = str(exc)
                    failed['judge_seconds'] = None
                    evaluated_results.append(failed)
                pbar.update(1)

    return evaluated_results


def evaluate_from_files(
    answers_file: str,
    test_file: str,
    judge_config: str,
    judge_server: str = "api",
    output_file: Optional[str] = None,
    max_workers: int = 1,
    usage_log: Optional[str] = None,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Standalone evaluation function to evaluate already-generated answers.

    Args:
        answers_file: Path to JSONL file with answers (output from run.py)
        test_file: Path to original test JSONL file
        judge_config: Path to judge configuration YAML file
        judge_server: Judge server type ("api" or "vllm")
        output_file: Path to save evaluation results (optional)

    Returns:
        Evaluation summary with statistics and detailed results
    """
    # Initialize judge client
    judge_client = ModelClient(config_path=judge_config, server_type=judge_server)
    if usage_log:
        judge_client.set_usage_log(usage_log, role="judge")
    print(f"✅ Initialized judge client: {judge_client.provider}/{judge_client.model}")

    # Load test data to get original information
    original_episodes = {}
    with open(test_file, 'r') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episode_id = episode_data.get("episode_id")
            original_episodes[episode_id] = episode_data

    # Load answers
    episode_results = []
    with open(answers_file, 'r') as f:
        for line in f:
            episode_results.append(json.loads(line.strip()))

    # Build QA results for evaluation
    all_qa_results = []
    for episode in episode_results:
        episode_id = episode['episode_id']
        answer_list = episode['answer_list']

        # Get original episode data
        original_episode = original_episodes.get(episode_id, {})
        task_type = original_episode.get('task_type', 'unknown')
        domain = original_episode.get('domain', 'unknown')
        task_description = original_episode.get('task', '')
        qa_pairs = original_episode.get('qa_pairs', [])

        # Match answers with golden answers
        for i, (predicted_answer, qa_pair) in enumerate(zip(answer_list, qa_pairs)):
            all_qa_results.append({
                'episode_id': episode_id,
                'qa_index': i,
                'task_type': task_type,
                'domain': domain,
                'task_description': task_description,
                'question': qa_pair.get('question', ''),
                'golden_answer': qa_pair.get('answer', ''),
                'predicted_answer': predicted_answer,
                'qa_type': qa_pair.get('type') or 'unknown',
            })

    # Evaluate using LLM judge
    print(f"\n🔍 Evaluating {len(all_qa_results)} QA pairs...")
    evaluated_results = evaluate_batch(
        qa_results=all_qa_results,
        judge_client=judge_client,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
    )

    # Calculate statistics by different dimensions
    stats_by_task_type = {}
    stats_by_domain = {}
    stats_by_qa_type = {}

    for r in evaluated_results:
        task_type = r.get('task_type', 'unknown')
        domain = r.get('domain', 'unknown')
        qa_type = r.get('qa_type', 'unknown')
        score = r.get('score')
        f1 = r.get('f1', 0.0)
        if score is None:
            continue

        # Group by task_type
        if task_type not in stats_by_task_type:
            stats_by_task_type[task_type] = []
        stats_by_task_type[task_type].append((score, f1))

        # Group by domain
        if domain not in stats_by_domain:
            stats_by_domain[domain] = []
        stats_by_domain[domain].append((score, f1))

        # Group by qa_type
        if qa_type not in stats_by_qa_type:
            stats_by_qa_type[qa_type] = []
        stats_by_qa_type[qa_type].append((score, f1))

    # Calculate averages
    task_type_stats = {
        k: {
            'count': len(v),
            'avg_score': sum(s for s, _ in v) / len(v) if v else 0,
            'accuracy': sum(1 for s, _ in v if s == 1.0) / len(v) if v else 0,
            'f1': sum(f1 for _, f1 in v) / len(v) if v else 0,
        }
        for k, v in stats_by_task_type.items()
    }

    domain_stats = {
        k: {
            'count': len(v),
            'avg_score': sum(s for s, _ in v) / len(v) if v else 0,
            'accuracy': sum(1 for s, _ in v if s == 1.0) / len(v) if v else 0,
            'f1': sum(f1 for _, f1 in v) / len(v) if v else 0,
        }
        for k, v in stats_by_domain.items()
    }

    qa_type_stats = {
        k: {
            'count': len(v),
            'avg_score': sum(s for s, _ in v) / len(v) if v else 0,
            'accuracy': sum(1 for s, _ in v if s == 1.0) / len(v) if v else 0,
            'f1': sum(f1 for _, f1 in v) / len(v) if v else 0,
        }
        for k, v in stats_by_qa_type.items()
    }

    # Build evaluation summary
    evaluation_summary = {
        'config': {
            'judge_provider': judge_client.provider,
            'judge_model': judge_client.model,
            'usage_log': usage_log,
        },
        'overall': {
            'total_questions': len(evaluated_results),
            'evaluated_questions': sum(1 for r in evaluated_results if r.get('score') is not None),
            'failed_questions': sum(1 for r in evaluated_results if r.get('score') is None),
            'avg_score': (
                sum(r['score'] for r in evaluated_results if r.get('score') is not None)
                / max(1, sum(1 for r in evaluated_results if r.get('score') is not None))
            ),
            'accuracy': (
                sum(1 for r in evaluated_results if r.get('score') == 1.0)
                / max(1, sum(1 for r in evaluated_results if r.get('score') is not None))
            ),
            'f1': sum(r.get('f1', 0.0) for r in evaluated_results) / len(evaluated_results) if evaluated_results else 0,
        },
        'by_task_type': task_type_stats,
        'by_domain': domain_stats,
        'by_qa_type': qa_type_stats,
        'results': evaluated_results,
    }

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        print(f"✅ Evaluation results saved to: {output_file}")

    return evaluation_summary


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    print(f"\n📊 Overall Performance:")
    print(f"  Total questions: {summary['overall']['total_questions']}")
    print(f"  Average score: {summary['overall']['avg_score']:.4f}")
    print(f"  Accuracy: {summary['overall']['accuracy']:.4f}")
    print(f"  F1: {summary['overall'].get('f1', 0.0):.4f}")

    print(f"\n🌐 By Domain:")
    for domain, stats in sorted(summary.get('by_domain', {}).items()):
        print(f"  {domain}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['count']} questions)")
        print(f"    F1: {stats.get('f1', 0.0):.4f}")

    print(f"\n❓ By QA Type:")
    for qa_type, stats in sorted(summary.get('by_qa_type', {}).items()):
        print(f"  Type {qa_type}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['count']} questions)")
        print(f"    F1: {stats.get('f1', 0.0):.4f}")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone LLM-as-Judge Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate answers using GPT judge
  python src/eval.py \\
    --answers-file results/answers_model_20260304_120000.jsonl \\
    --test-file dataset/test/mcq_set.jsonl \\
    --judge-config configs/llm_judge.yaml \\
    --judge-server api \\
    --output-file results/evaluation_results.json

  # Evaluate answers using Qwen judge
  python src/eval.py \\
    --answers-file results/answers_model_20260304_120000.jsonl \\
    --test-file dataset/test/mcq_set.jsonl \\
    --judge-config configs/llm_judge.yaml \\
    --judge-server vllm \\
    --output-file results/evaluation_results.json
        """
    )

    parser.add_argument("--answers-file", type=str, required=True,
                        help="Path to JSONL file with generated answers")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to original test JSONL file")
    parser.add_argument("--judge-config", type=str, required=True,
                        help="Path to judge LLM configuration YAML file")
    parser.add_argument("--judge-server", type=str, choices=["api", "vllm"], default="api",
                        help="Judge server type (api or vllm)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation results JSON file")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of concurrent judge calls")
    parser.add_argument("--usage-log", type=str, default=None,
                        help="Optional JSONL path for judge API usage logging")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep partial evaluation results if a judge call fails")

    args = parser.parse_args()

    # Validate files exist
    if not Path(args.answers_file).exists():
        parser.error(f"Answers file not found: {args.answers_file}")
    if not Path(args.test_file).exists():
        parser.error(f"Test file not found: {args.test_file}")
    if not Path(args.judge_config).exists():
        parser.error(f"Judge config not found: {args.judge_config}")

    # Run evaluation
    summary = evaluate_from_files(
        answers_file=args.answers_file,
        test_file=args.test_file,
        judge_config=args.judge_config,
        judge_server=args.judge_server,
        output_file=args.output_file,
        max_workers=args.max_workers,
        usage_log=args.usage_log,
        continue_on_error=args.continue_on_error,
    )

    # Print summary
    print_evaluation_summary(summary)
