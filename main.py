import os
import csv
import time
import argparse
import pandas as pd

from src.config import (
    DEFAULT_CSV_INPUT, DEFAULT_CSV_OUTPUT,
    DEFAULT_DEVICE, QWEN_DEVICE, VIDEOLLAMA_DEVICE, EMBEDDING_DEVICE,
    MAX_ITERATIONS, MAX_OUTER_LOOPS, SAFETY_THRESHOLD, HAILOU_API_KEY
)
from src.models import initialize_qwen_model, initialize_embedding_model, initialize_videollama3
from src.video_generators import (
    generate_video_with_veo2, generate_video_with_sora2,
    generate_video_with_kling, generate_video_with_luma, generate_video_with_hailou,
    initialize_luma_client
)
from src.evaluators import evaluate_video_safety, evaluate_scene_influence
from src.utils import extract_scenes
from src.prompts import generate_scene_prompt, modify_scene_prompt
from src.strategies import save_strategy_to_library


def main():
    parser = argparse.ArgumentParser(description='Iterative Attack Generator')
    parser.add_argument('--category', type=int, nargs='+', default=None,
                       help='Category numbers (e.g., 1 3 5). If not specified, process all categories')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                       help=f'GPU device (default: {DEFAULT_DEVICE})')
    parser.add_argument('--csv-input', type=str, default=str(DEFAULT_CSV_INPUT),
                       help='Input CSV file path')
    parser.add_argument('--csv-output', type=str, default=str(DEFAULT_CSV_OUTPUT),
                       help='Output CSV file path')
    parser.add_argument('--video-generator', type=str, choices=['veo2', 'luma', 'sora2', 'kling', 'hailou'], default='sora2',
                       help='Video generator selection (default: sora2)')
    parser.add_argument('--use-videollama3', action='store_true',
                       help='Use VideoLLaMA3 for evaluation (default: GPT4o)')
    parser.add_argument('--num-frames', type=int, default=5,
                       help='Number of frames for GPT4o evaluation (default: 5)')
    parser.add_argument('--scale-percent', type=int, default=20,
                       help='Frame size reduction percentage for GPT4o evaluation (default: 20)')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                       help=f'Maximum iterations (default: {MAX_ITERATIONS})')
    parser.add_argument('--max-outer-loops', type=int, default=MAX_OUTER_LOOPS,
                       help=f'Maximum outer loops (default: {MAX_OUTER_LOOPS})')
    
    args = parser.parse_args()
    
    max_iterations = args.max_iterations
    max_outer_loops = args.max_outer_loops
    
    device = args.device
    print(f"Using device: {device}")
    
    video_generator = args.video_generator
    print(f"Video generator: {video_generator.upper()}")
    
    use_gpt4o = not args.use_videollama3
    print(f"Evaluation model: {'GPT4o' if use_gpt4o else 'VideoLLaMA3'}")
    if use_gpt4o:
        print(f"GPT4o settings - frames: {args.num_frames}, scale: {args.scale_percent}%")
    
    target_categories = args.category
    if target_categories is not None:
        print(f"Processing categories: {target_categories}")
    else:
        print("Processing all categories")
    
    main_device = args.device
    qwen_device = QWEN_DEVICE
    videollama_device = VIDEOLLAMA_DEVICE
    embedding_device = EMBEDDING_DEVICE
    
    if max_iterations == 1 and max_outer_loops == 1:
        print("Warning: max_iterations=1 and max_outer_loops=1 - skipping VideoLLaMA3, Qwen, and Embedding initialization")
        videollama_model = None
        videollama_processor = None
        qwen_model = None
        qwen_tokenizer = None
        embedding_model = None
    else:
        print(f"GPU device settings:")
        print(f"   Main device: {main_device}")
        
        print(f"   VideoLLaMA3: {videollama_device}")
        videollama_model, videollama_processor = initialize_videollama3(videollama_device)
        
        qwen_model = None
        qwen_tokenizer = None
        embedding_model = None
        
        if max_outer_loops > 1:
            print("Initializing models for strategy save/load...")
            print(f"   Qwen model: {qwen_device}")
            print(f"   Embedding model: {embedding_device}")
            qwen_model, qwen_tokenizer = initialize_qwen_model(qwen_device)
            embedding_model = initialize_embedding_model(embedding_device)
            print("Strategy-related models initialized")
        else:
            print("Warning: max_outer_loops=1, disabling strategy save/load")
            print("   Not loading Qwen and Embedding models")
    
    luma_client = None
    if video_generator == 'luma':
        luma_client = initialize_luma_client()
    
    hailou_api_key = None
    if video_generator == 'hailou':
        hailou_api_key = HAILOU_API_KEY
        if not hailou_api_key:
            raise ValueError("HAILOU_API_KEY environment variable is required for Hailou generator")
    
    csv_file = args.csv_input
    df = pd.read_csv(csv_file)
    print(f"Loaded input CSV file: {csv_file}")
    
    csv_output_file = args.csv_output
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
    
    csv_headers = ['base_prompt']
    for outer_loop in range(1, max_outer_loops + 1):
        for iteration in range(1, max_iterations + 1):
            csv_headers.extend([
                f'step_{outer_loop}_{iteration}', f'outer_loop_{outer_loop}_{iteration}', f'step_type_{outer_loop}_{iteration}',
                f'scene_prompt_{outer_loop}_{iteration}', f'video_path_{outer_loop}_{iteration}',
                f'safety_score_{outer_loop}_{iteration}', f'attack_result_{outer_loop}_{iteration}',
                f'most_influential_scene_{outer_loop}_{iteration}', f'scenes_{outer_loop}_{iteration}'
            ])
    
    if not os.path.exists(csv_output_file):
        with open(csv_output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    for idx, row in df.iterrows():
        base_prompt = row['base']
        category = row['category'] if 'category' in row else None
        
        if target_categories is not None and category not in target_categories:
            print(f"\n{'='*50}")
            print(f"Skipping prompt {idx+1}/{len(df)}: Category {category} (targets: {target_categories})")
            print(f"{'='*50}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing prompt {idx+1}/{len(df)}: {base_prompt}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        
        prompt_results = {
            'base_prompt': base_prompt
        }
        
        used_strategy_ids = set()
        
        outer_loop_count = 0
        attack_successful = False
        
        while outer_loop_count < max_outer_loops and not attack_successful:
            outer_loop_count += 1
            print(f"\nOuter loop {outer_loop_count}/{max_outer_loops} started")
            
            if outer_loop_count > 1:
                print(f"Outer Loop {outer_loop_count}: Searching for new strategy from library")
            
            current_scene_prompt = None
            safety_category = "Safe"
            conversation_history = []
            most_influential_scene = "N/A"
            reward_info = None
            
            iteration = 1
            
            while iteration <= max_iterations and not attack_successful:
                print(f"\n--- Iteration {iteration} (outer loop {outer_loop_count}) ---")
                
                if iteration == 1:
                    if outer_loop_count > 1:
                        print(f"Outer Loop {outer_loop_count}: Searching for new strategy from library...")
                        step_type = "strategy_based"
                    else:
                        print("Generating initial scene prompt...")
                        step_type = "initial"
                    
                    current_scene_prompt, conversation_history, used_strategy_id = generate_scene_prompt(
                        base_prompt,
                        conversation_history=None,
                        qwen_model=qwen_model,
                        qwen_tokenizer=qwen_tokenizer,
                        embedding_model=embedding_model,
                        device=qwen_device,
                        used_strategy_ids=used_strategy_ids
                    )
                    
                    if used_strategy_id:
                        used_strategy_ids.add(used_strategy_id)
                        print(f"Strategy {used_strategy_id} used, added to tracking list")
                else:
                    print("Modifying scene prompt...")
                    current_scene_prompt, conversation_history = modify_scene_prompt(
                        base_prompt,
                        extract_scenes(current_scene_prompt),
                        most_influential_scene,
                        safety_category,
                        conversation_history,
                        reward_info
                    )
                    step_type = "modified"
                
                if not current_scene_prompt:
                    print("Scene prompt generation failed, moving to next iteration")
                    prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                    prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                    prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                    prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = "FAILED"
                    prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = "FAILED"
                    prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = 0.0
                    prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'FAILED'
                    prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = "N/A"
                    prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = "[]"
                    
                    iteration += 1
                    continue
                
                print(f"Generated scene prompt:\n{current_scene_prompt}")
                
                try:
                    if video_generator == 'veo2':
                        print("Generating video with Veo2...")
                        video_path = generate_video_with_veo2(current_scene_prompt)
                    elif video_generator == 'luma':
                        print("Generating video with Luma...")
                        video_path = generate_video_with_luma(current_scene_prompt, luma_client)
                    elif video_generator == 'sora2':
                        print("Generating video with Sora2...")
                        video_path = generate_video_with_sora2(current_scene_prompt)
                    elif video_generator == 'kling':
                        print("Generating video with Kling AI...")
                        video_path = generate_video_with_kling(current_scene_prompt)
                    elif video_generator == 'hailou':
                        print("Generating video with Hailou...")
                        video_path = generate_video_with_hailou(current_scene_prompt, hailou_api_key)
                    
                    if video_path == "GUARDRAIL_BLOCKED":
                        if video_generator == 'hailou':
                            generator_name = "Hailou"
                        else:
                            generator_name = video_generator.upper()
                        print(f"{generator_name} guardrail blocked, retrying with random scene selection")
                        
                        scenes = extract_scenes(current_scene_prompt)
                        most_influential_scene = "GUARDRAIL_BLOCKED"
                        reward_info = {"guardrail_blocked_videos_error": True}
                        
                        prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                        prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                        prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                        prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = current_scene_prompt
                        prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = "GUARDRAIL_BLOCKED"
                        prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = 0.0
                        prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'GUARDRAIL_BLOCKED'
                        prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = most_influential_scene
                        prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = str(scenes)
                        
                        iteration += 1
                        continue
                    
                    if not video_path or not os.path.exists(video_path):
                        if video_generator == 'hailou':
                            generator_name = "Hailou"
                        else:
                            generator_name = video_generator.upper()
                        print(f"{generator_name} generation failed, moving to next iteration")
                        prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                        prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                        prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                        prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = current_scene_prompt
                        prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = "FAILED"
                        prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = 0.0
                        prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'FAILED'
                        prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = "N/A"
                        prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = str(extract_scenes(current_scene_prompt))
                        
                        iteration += 1
                        continue
                        
                except Exception as e:
                    if video_generator == 'hailou':
                        generator_name = "Hailou"
                    else:
                        generator_name = video_generator.upper()
                    print(f"{generator_name} generation error: {e}, moving to next iteration")
                    prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                    prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                    prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                    prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = current_scene_prompt
                    prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = f"ERROR: {str(e)}"
                    prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = 0.0
                    prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'ERROR'
                    prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = "N/A"
                    prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = str(extract_scenes(current_scene_prompt))
                    
                    iteration += 1
                    continue
                
                if video_generator == 'hailou':
                    generator_name = "Hailou"
                else:
                    generator_name = video_generator.upper()
                print(f"Generated {generator_name}: {video_path}")
                
                print("Evaluating safety...")
                safety_score = evaluate_video_safety(
                    videollama_model, videollama_processor, video_path, base_prompt, videollama_device, category,
                    use_gpt4o=use_gpt4o, num_frames=args.num_frames, scale_percent=args.scale_percent
                )
                
                scenes = extract_scenes(current_scene_prompt)
                if safety_score < SAFETY_THRESHOLD and videollama_model is not None and videollama_processor is not None:
                    most_influential_scene = evaluate_scene_influence(
                        videollama_model, videollama_processor, video_path, scenes, videollama_device
                    )
                elif safety_score < SAFETY_THRESHOLD:
                    print("Warning: VideoLLaMA3 is not initialized, skipping scene influence evaluation")
                    most_influential_scene = "N/A"
                else:
                    most_influential_scene = "N/A"
                
                print(f"Safety score: {safety_score:.3f}")
                print(f"Most influential scene: {most_influential_scene}")
                
                if safety_score >= SAFETY_THRESHOLD:
                    safety_category = "Unsafe"
                    print(f"Attack successful! Safety score: {safety_score:.3f}")
                    attack_successful = True
                    
                    if max_outer_loops > 1:
                        print(f"Attack successful! Saving strategy... (Outer Loop: {outer_loop_count}, Iteration: {iteration})")
                        scenes_list = extract_scenes(current_scene_prompt)
                        strategy_info = {
                            "outer_loop": outer_loop_count,
                            "iteration": iteration,
                            "step_type": step_type,
                            "safety_score": safety_score,
                            "most_influential_scene": most_influential_scene,
                            "video_path": video_path,
                            "is_oneshot": False,
                            "scene_manipulation_info": {
                                "manipulated_scene": most_influential_scene,
                                "manipulation_iteration": iteration,
                                "original_scenes": scenes_list,
                                "final_scene_prompt": current_scene_prompt
                            }
                        }
                        
                        strategy_id = save_strategy_to_library(
                            base_prompt,
                            current_scene_prompt,
                            scenes_list,
                            strategy_info,
                            qwen_model,
                            qwen_tokenizer,
                            qwen_device
                        )
                        
                        if strategy_id:
                            print(f"Strategy saved: {strategy_id}")
                        else:
                            print("Strategy save failed")
                    else:
                        print(f"Attack successful! (Outer Loop: {outer_loop_count}, Iteration: {iteration})")
                        print("Warning: max_outer_loops=1, skipping strategy save")
                    
                    prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                    prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                    prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                    prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = current_scene_prompt
                    prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = video_path
                    prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = safety_score
                    prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'Unsafe'
                    prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = most_influential_scene
                    prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = str(scenes)
                    
                    break
                else:
                    safety_category = "Safe"
                    print(f"Safe (score: {safety_score:.3f}), retrying with scene modification")
                    
                    prompt_results[f'step_{outer_loop_count}_{iteration}'] = iteration
                    prompt_results[f'outer_loop_{outer_loop_count}_{iteration}'] = outer_loop_count
                    prompt_results[f'step_type_{outer_loop_count}_{iteration}'] = step_type
                    prompt_results[f'scene_prompt_{outer_loop_count}_{iteration}'] = current_scene_prompt
                    prompt_results[f'video_path_{outer_loop_count}_{iteration}'] = video_path
                    prompt_results[f'safety_score_{outer_loop_count}_{iteration}'] = safety_score
                    prompt_results[f'attack_result_{outer_loop_count}_{iteration}'] = 'Safe'
                    prompt_results[f'most_influential_scene_{outer_loop_count}_{iteration}'] = most_influential_scene
                    prompt_results[f'scenes_{outer_loop_count}_{iteration}'] = str(scenes)
                
                iteration += 1
                
                if iteration <= max_iterations:
                    print("Waiting 3 seconds before next iteration...")
                    time.sleep(3)
            
            if attack_successful:
                print(f"Outer loop {outer_loop_count} attack successful!")
                break
            else:
                print(f"Outer loop {outer_loop_count} attack failed ({max_iterations} iterations completed)")
                if outer_loop_count < max_outer_loops:
                    print(f"Restarting with new initial scene prompt...")
                    time.sleep(5)
                else:
                    print(f"Reached maximum outer loop count ({max_outer_loops})")
        
        csv_row = []
        for header in csv_headers:
            if header in prompt_results:
                csv_row.append(prompt_results[header])
            else:
                csv_row.append('')
        
        with open(csv_output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
        
        print(f"Results saved to CSV: {csv_output_file}")
    
    print("\nAll prompts processed!")


if __name__ == "__main__":
    main()
