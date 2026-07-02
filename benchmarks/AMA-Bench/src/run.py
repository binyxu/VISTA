import argparse
import yaml
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory_interface import MemoryQAInterface
from src.model_client import ModelClient
from src.method_register import list_methods
from src.evaluate import evaluate_batch, print_evaluation_summary
from utils.evaluation_metrics import compute_f1_score
from utils.embedding import EmbeddingEngine


def save_answers_jsonl(path: Path, episode_results):
    with open(path, 'w') as f:
        for episode in sorted(episode_results, key=lambda x: x['episode_id']):
            f.write(json.dumps(episode) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="AMA-Bench Memory QA with LLM-as-Judge Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with LLM-as-judge evaluation (default)
  python src/run.py \\
    --llm-server api \\
    --llm-config configs/gpt-5.2.yaml \\
    --judge-config configs/llm_judge.yaml \\
    --subset mcq \\
    --method longcontext

  # Run without evaluation
  python src/run.py \\
    --llm-server vllm \\
    --llm-config configs/qwen3-32B.yaml \\
    --judge-config configs/llm_judge.yaml \\
    --subset mcq \\
    --evaluate False
        """
    )
    
    # LLM Server configuration
    parser.add_argument("--llm-server", type=str, required=True,
                        choices=["api", "vllm"],
                        help="LLM server type: 'api' for API-based models, 'vllm' for VLLM server")
    parser.add_argument("--llm-config", type=str, required=True,
                        help="Path to LLM configuration YAML file")

    # Dataset configuration
    parser.add_argument("--subset", type=str, required=True,
                        choices=["mcq", "openend"],
                        help="Dataset subset: 'mcq' or 'openend'")

    # Baseline/Method configuration
    parser.add_argument("--method", type=str, default="longcontext",
                        help=f"Memory method. Available: {', '.join(list_methods())}. Default: longcontext")

    parser.add_argument("--method-config", type=str, default=None,
                        help="Method-specific configuration as JSON string or path to JSON/YAML file")

    # Test configuration
    parser.add_argument("--test-dir", type=str, default="dataset/test",
                        help="Directory containing test files. Default: dataset/test")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to test JSONL file. Auto-determined from test-dir and subset if not specified")

    # Concurrency configuration
    parser.add_argument("--max-concurrency-episodes", type=int, default=1,
                        help="Maximum number of episodes to process concurrently. Default: 1")
    parser.add_argument("--max-concurrency-questions-per-episode", type=int, default=1,
                        help="Maximum number of questions per episode to process concurrently. Default: 1")

    # Judge configuration
    parser.add_argument("--judge-config", type=str, default="configs/llm_judge.yaml",
                        help="Path to judge LLM configuration YAML file. Required when --evaluate True")
    parser.add_argument("--judge-server", type=str, choices=["api", "vllm"], default="api",
                        help="Judge server type (api or vllm). Default: api")
    parser.add_argument("--judge-max-concurrency", type=int, default=None,
                        help="Max concurrent judge workers. Default: max-concurrency-episodes * max-concurrency-questions-per-episode")

    # Evaluation flag
    parser.add_argument("--evaluate", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to evaluate answers after generation. Default: True")

    # Sampling / filtering configuration
    parser.add_argument("--samples", type=int, default=None,
                        help="Randomly sample N episodes from the dataset. Mutually exclusive with --domains / --episode-ids.")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated list of domains to evaluate (e.g. 'embodied_ai,software_engineer'). "
                             "Mutually exclusive with --samples / --episode-ids.")
    parser.add_argument("--episode-ids", type=str, default=None,
                        help="Comma-separated list of episode_id values to evaluate (e.g. '30,31,32'). "
                             "Mutually exclusive with --samples / --domains.")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results. Default: results")
    parser.add_argument("--usage-log", type=str, default=None,
                        help="JSONL file for per-API-call usage/timing records. Default: output_dir/usage_<run>.jsonl")
    parser.add_argument("--skip-existing-answers", type=str, default=None,
                        help="Optional answers_*.jsonl file. Episodes already present there are skipped and reused.")

    args = parser.parse_args()

    # Validate mutual exclusivity of --samples / --domains / --episode-ids
    _filters_set = sum(
        x is not None for x in (args.samples, args.domains, args.episode_ids)
    )
    if _filters_set > 1:
        parser.error("--samples, --domains, and --episode-ids are mutually exclusive.")

    # Auto-configure test file based on test_dir and subset
    if args.test_file is None:
        test_dir = Path(args.test_dir)
        if args.subset == "mcq":
            args.test_file = str(test_dir / "mcq_set.jsonl")
        elif args.subset == "openend":
            args.test_file = str(test_dir / "open_end_qa_set.jsonl")

    if not Path(args.test_file).exists():
        parser.error(f"Test file not found: {args.test_file}")
    if args.evaluate and not Path(args.judge_config).exists():
        parser.error(f"Judge config not found: {args.judge_config}")
    if args.output_dir == "results":
        args.output_dir = f"results/{args.subset}"

    # Validate method
    available_methods = list_methods()
    if args.method not in available_methods:
        parser.error(f"Invalid method '{args.method}'. Available methods: {', '.join(available_methods)}")

    # For longcontext, default method_config to llm_config so model length settings are picked up
    if args.method == "longcontext" and not args.method_config:
        args.method_config = args.llm_config

    # For ama_agent, auto-load configs/ama_agent.yaml (embedding engine config lives there)
    if args.method == "ama_agent" and not args.method_config:
        default_ama_config = "configs/ama_agent.yaml"
        if Path(default_ama_config).exists():
            args.method_config = default_ama_config
            print(f"Auto-using AMA-Agent config: {default_ama_config}")

    # Create main LLM client
    client = ModelClient(config_path=args.llm_config, server_type=args.llm_server)
    print(f"✅ Initialized LLM client: {client.provider}/{client.model}")

    # Create judge client (only needed for evaluation)
    judge_client = None
    if args.evaluate:
        judge_client = ModelClient(config_path=args.judge_config, server_type=args.judge_server)
        print(f"✅ Initialized judge client: {judge_client.provider}/{judge_client.model}")

    # Initialize embedding engine from method config
    embedding_engine = None
    if args.method_config:
        try:
            # Load method config to check for embedding_engine settings
            method_config_path = Path(args.method_config)
            if method_config_path.exists():
                if method_config_path.suffix in ['.yaml', '.yml']:
                    with open(method_config_path, 'r') as f:
                        method_config_data = yaml.safe_load(f)
                else:
                    with open(method_config_path, 'r') as f:
                        method_config_data = json.load(f)

                # Check if embedding_engine is configured
                embedding_config = method_config_data.get('embedding_engine')
                if embedding_config and embedding_config is not None:
                    embedding_engine = EmbeddingEngine(
                        model_name=embedding_config.get('model_name'),
                        base_url=(
                            embedding_config.get('base_url')
                            or os.getenv('VENUS_EMBED_BASE_URL')
                        ),
                        api_key=(
                            embedding_config.get('api_key')
                            or os.getenv('VENUS_API_KEY')
                            or os.getenv('OPENAI_API_KEY')
                            or os.getenv('LOCA_OPENAI_API_KEY')
                            or 'EMPTY'
                        ),
                        batch_size=embedding_config.get('batch_size', 8),
                        max_length=embedding_config.get('max_length', 512),
                        auto_launch=embedding_config.get('auto_launch', False),
                        host=embedding_config.get('host', '127.0.0.1'),
                        port=embedding_config.get('port', 8003),
                        runner=embedding_config.get('runner', 'pooling'),
                        cuda_visible_devices=embedding_config.get('cuda_visible_devices'),
                        tensor_parallel_size=embedding_config.get('tensor_parallel_size', 1),
                        gpu_memory_utilization=embedding_config.get('gpu_memory_utilization', 0.9),
                        startup_timeout=embedding_config.get('startup_timeout', 120),
                    )
                    print(f"✅ Initialized embedding engine: {embedding_config.get('model_name')}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize embedding engine: {e}")

    # Register shutdown hook so the embedding server is stopped on any exit path
    if embedding_engine is not None:
        import atexit
        atexit.register(embedding_engine.shutdown)

    # Load and filter episodes (for --samples / --domains / --episode-ids)
    filtered_episodes = None
    if args.samples is not None or args.domains is not None or args.episode_ids is not None:
        import random
        all_episodes = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_episodes.append(json.loads(line.strip()))

        if args.episode_ids is not None:
            target_ids = {str(i).strip() for i in args.episode_ids.split(',') if str(i).strip()}
            filtered_episodes = [ep for ep in all_episodes if str(ep.get('episode_id', '')) in target_ids]
            missing = target_ids - {str(ep.get('episode_id', '')) for ep in filtered_episodes}
            if missing:
                parser.error(f"--episode-ids referenced ids not found in test file: {sorted(missing)}")
            print(f"Filtering by episode-ids {sorted(target_ids)}: {len(all_episodes)} → {len(filtered_episodes)} episodes")
        elif args.domains is not None:
            target_domains = {d.strip() for d in args.domains.split(',')}
            filtered_episodes = [ep for ep in all_episodes if ep.get('domain', '') in target_domains]
            print(f"Filtering by domains {target_domains}: {len(all_episodes)} → {len(filtered_episodes)} episodes")
        elif args.samples is not None:
            if args.samples >= len(all_episodes):
                filtered_episodes = all_episodes
                print(f"--samples {args.samples} >= total episodes {len(all_episodes)}, using all episodes")
            else:
                filtered_episodes = random.sample(all_episodes, args.samples)
                print(f"Randomly sampled {args.samples} episodes from {len(all_episodes)} total")

    # Create interface
    interface = MemoryQAInterface(
        client=client,
        method_name=args.method,
        method_config=args.method_config,
        max_concurrency_episodes=args.max_concurrency_episodes,
        max_concurrency_questions=args.max_concurrency_questions_per_episode,
        subset=args.subset,
        embedding_engine=embedding_engine,
    )

    # Prepare output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = client.model.replace("/", "_")
    subset_suffix = f"_{args.subset}" 
    method_suffix = f"_{args.method}" 
    base_filename = f"{model_name}{subset_suffix}{method_suffix}_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answers_path = output_dir / f"answers_{base_filename}.jsonl"
    results_path = output_dir / f"results_{base_filename}.json"
    usage_path = Path(args.usage_log) if args.usage_log else output_dir / f"usage_{base_filename}.jsonl"
    if usage_path.exists():
        usage_path.unlink()
    client.set_usage_log(str(usage_path), role="llm")
    if judge_client is not None:
        judge_client.set_usage_log(str(usage_path), role="judge")
    if embedding_engine is not None:
        embedding_engine.set_usage_log(str(usage_path))
    print(f"Usage log: {usage_path}")

    existing_episode_results = []
    existing_episode_ids = set()
    if args.skip_existing_answers:
        skip_path = Path(args.skip_existing_answers)
        if skip_path.exists():
            with open(skip_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    existing_episode_results.append(episode)
                    existing_episode_ids.add(str(episode.get('episode_id')))
            if filtered_episodes is None:
                parser.error("--skip-existing-answers currently requires explicit filtering via --episode-ids/--samples/--domains")
            before = len(filtered_episodes)
            filtered_episodes = [
                ep for ep in filtered_episodes
                if str(ep.get('episode_id')) not in existing_episode_ids
            ]
            print(f"Reusing {len(existing_episode_results)} existing episodes from {skip_path}; remaining {before} → {len(filtered_episodes)}")
        else:
            parser.error(f"--skip-existing-answers not found: {skip_path}")

    # Phase 1: Generate answers
    print("\n" + "="*70)
    print("PHASE 1: GENERATING ANSWERS")

    episode_results = list(existing_episode_results)

    def checkpoint_episode(result):
        episode_results.append(result)
        save_answers_jsonl(answers_path, episode_results)

    try:
        interface.run(
            file_path=args.test_file,
            episodes=filtered_episodes,
            on_episode_result=checkpoint_episode,
        )
    finally:
        if episode_results:
            save_answers_jsonl(answers_path, episode_results)
            print(f"\n💾 Partial answers checkpoint saved to: {answers_path}")

    # Save answers to JSONL
    save_answers_jsonl(answers_path, episode_results)
    print(f"\n✅ Answers saved to: {answers_path}")
    print(f"   Total episodes processed: {len(episode_results)}")

    # Phase 2: Evaluate answers (if enabled)
    if not args.evaluate:
        print("\n⏭️ Skipping evaluation (--evaluate False)")
        return

    print("\n" + "="*70)
    print("PHASE 2: EVALUATING ANSWERS WITH LLM-AS-JUDGE")
    print("="*70)

    # Load original test data to get golden answers and metadata
    original_episodes = {}
    with open(args.test_file, 'r') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episode_id = episode_data.get("episode_id")
            original_episodes[episode_id] = episode_data

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
            golden_answer = qa_pair.get('answer', '')
            all_qa_results.append({
                'episode_id': episode_id,
                'qa_index': i,
                'task_type': task_type,
                'domain': domain,
                'task_description': task_description,
                'question': qa_pair.get('question', ''),
                'golden_answer': golden_answer,
                'predicted_answer': predicted_answer,
                'f1': compute_f1_score(predicted_answer, golden_answer),
                'qa_type': qa_pair.get('type') or 'unknown',
            })

    # Evaluate using LLM judge
    judge_max_concurrency = args.judge_max_concurrency or (
        args.max_concurrency_episodes * args.max_concurrency_questions_per_episode
    )
    print(f"\n🔍 Evaluating {len(all_qa_results)} QA pairs... (judge concurrency: {judge_max_concurrency})")
    evaluated_results = evaluate_batch(
        qa_results=all_qa_results,
        judge_client=judge_client,
        max_workers=judge_max_concurrency,
    )

    # Calculate statistics by different dimensions
    stats_by_task_type = {}
    stats_by_domain = {}
    stats_by_qa_type = {}

    for r in evaluated_results:
        task_type = r.get('task_type', 'unknown')
        domain = r.get('domain', 'unknown')
        qa_type = r.get('qa_type', 'unknown')
        score = r['score']
        f1 = r.get('f1', 0.0)

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

    # Save evaluation results with detailed statistics
    evaluation_summary = {
        'config': {
            'provider': client.provider,
            'model': client.model,
            'method': args.method,
            'subset': args.subset,
            'judge_provider': judge_client.provider,
            'judge_model': judge_client.model,
            'timestamp': timestamp,
            'usage_log': str(usage_path),
        },
        'overall': {
            'total_questions': len(evaluated_results),
            'avg_score': sum(r['score'] for r in evaluated_results) / len(evaluated_results) if evaluated_results else 0,
            'accuracy': sum(1 for r in evaluated_results if r['score'] == 1.0) / len(evaluated_results) if evaluated_results else 0,
            'f1': sum(r.get('f1', 0.0) for r in evaluated_results) / len(evaluated_results) if evaluated_results else 0,
        },
        'by_task_type': task_type_stats,
        'by_domain': domain_stats,
        'by_qa_type': qa_type_stats,
        'results': evaluated_results,
    }

    with open(results_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"\n✅ Evaluation results saved to: {results_path}")

    # Print summary
    print_evaluation_summary(evaluation_summary)


if __name__ == "__main__":
    main()
