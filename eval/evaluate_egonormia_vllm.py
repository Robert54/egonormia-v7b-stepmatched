#!/usr/bin/env python3
"""Fast EgoNormia MCQ evaluation using vLLM batch inference."""

import argparse
import ast
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from egonormia_prompts import (
    REASONING_PROMPT,
    V6_ACTION_PROMPT_TEMPLATE,
    V6_JUSTIFICATION_PROMPT_TEMPLATE,
    V6_SENSIBILITY_PROMPT_TEMPLATE,
)
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = "You are a helpful assistant."


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def has_thinking(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", text, flags=re.DOTALL))


def extract_think_block(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_answer(text: str):
    text = strip_thinking(text)
    leading = re.match(r"\s*([1-5])(?:\b|[.)])", text)
    if leading:
        return int(leading.group(1))
    decision_matches = re.findall(
        r"\b(?:choose|chose|select|selected|pick|picked)\s*(?:option\s*)?([1-5])\b",
        text,
        flags=re.IGNORECASE,
    )
    if decision_matches:
        return int(decision_matches[-1])
    anchored = re.search(
        r"(?im)^\s*(?:final answer|answer|choice)\s*[:\-]\s*([1-5])(?:\b|[.)])",
        text,
    )
    if anchored:
        return int(anchored.group(1))
    single_line_number = re.findall(r"(?m)^\s*([1-5])\s*$", text)
    if len(single_line_number) == 1:
        return int(single_line_number[0])
    trailing = re.search(r"(?m)^\s*([1-5])\.\s+\S+", text.rstrip().rsplit("\n", 1)[-1])
    if trailing:
        return int(trailing.group(1))
    return None


def parse_sensibility_answer(text: str):
    text = strip_thinking(text)
    anchored = re.match(r"\s*(\[[^\]]*\])", text, flags=re.DOTALL)
    candidates = [anchored.group(1)] if anchored else []
    if not candidates:
        candidates = re.findall(r"\[[^\]]*\]", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, list):
            continue
        normalized = []
        valid = True
        for item in parsed:
            if isinstance(item, bool):
                valid = False
                break
            if isinstance(item, (int, np.integer)):
                idx = int(item)
            elif isinstance(item, float) and item.is_integer():
                idx = int(item)
            else:
                valid = False
                break
            if idx < 1 or idx > 5:
                valid = False
                break
            normalized.append(idx)
        if valid:
            return sorted(set(normalized))
    return None


def iou(pred, gt):
    union = pred | gt
    if not union:
        return 1.0
    return len(pred & gt) / len(union)


def format_options(options):
    return "\n".join(f"{i + 1}. {opt if opt else ''}" for i, opt in enumerate(options))


def bootstrap_accuracy_ci(correct, n_resamples=1000, confidence=0.95, seed=42):
    if not correct:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(correct, dtype=np.float64)
    n = len(arr)
    accuracy = float(arr.mean())
    boot_accs = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_accs[i] = arr[rng.randint(0, n, size=n)].mean()
    alpha = 1.0 - confidence
    return (
        accuracy,
        float(np.percentile(boot_accs, 100 * alpha / 2)),
        float(np.percentile(boot_accs, 100 * (1 - alpha / 2))),
    )


def bootstrap_mean_ci(values, n_resamples=1000, confidence=0.95, seed=42):
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_means[i] = arr[rng.randint(0, n, size=n)].mean()
    alpha = 1.0 - confidence
    return (
        mean,
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def build_vllm_input(processor, video_path, prompt_text, nframes=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}", "nframes": nframes},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    return {
        "prompt": text_prompt,
        "multi_modal_data": {"video": video_inputs},
        "mm_processor_kwargs": video_kwargs,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast EgoNormia eval via vLLM")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test-path", dest="test_path", type=str, required=True)
    parser.add_argument("--video-base", dest="video_base", type=str, required=True)
    parser.add_argument("--taxonomy-path", dest="taxonomy_path", type=str, required=True)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256)
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=8192)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Append REASONING_PROMPT to each task prompt",
    )
    parser.add_argument("--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.test_path) as f:
        test_data = json.load(f)
    with open(args.taxonomy_path) as f:
        taxonomy = json.load(f)

    print(f"Loaded {len(test_data)} test samples")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    batch_inputs = []
    batch_meta = []

    for si, entry in enumerate(test_data):
        sid = entry["id"]
        video_path = os.path.join(args.video_base, entry["video"])
        if not os.path.exists(video_path):
            continue

        meta = taxonomy.get(sid, {})
        behaviors = meta.get("behaviors", [""] * 5)
        justifications = meta.get("justifications", [""] * 5)
        gt_action = int(meta.get("correct", 0)) + 1
        behavior_text = behaviors[gt_action - 1] if 1 <= gt_action <= len(behaviors) else ""

        action_prompt = V6_ACTION_PROMPT_TEMPLATE.format(
            behavior_options=format_options(behaviors)
        )
        if args.enable_thinking:
            action_prompt = action_prompt + "\n\n" + REASONING_PROMPT
        batch_inputs.append(build_vllm_input(processor, video_path, action_prompt, args.nframes))
        batch_meta.append((si, "action"))

        just_prompt = V6_JUSTIFICATION_PROMPT_TEMPLATE.format(
            behavior=behavior_text if behavior_text else "An action",
            justification_options=format_options(justifications),
        )
        if args.enable_thinking:
            just_prompt = just_prompt + "\n\n" + REASONING_PROMPT
        batch_inputs.append(build_vllm_input(processor, video_path, just_prompt, args.nframes))
        batch_meta.append((si, "justification"))

        sens_prompt = V6_SENSIBILITY_PROMPT_TEMPLATE.format(
            behavior_options=format_options(behaviors),
        )
        if args.enable_thinking:
            sens_prompt = sens_prompt + "\n\n" + REASONING_PROMPT
        batch_inputs.append(build_vllm_input(processor, video_path, sens_prompt, args.nframes))
        batch_meta.append((si, "sensibility"))

    print(f"Built {len(batch_inputs)} prompts ({len(batch_inputs) // 3} samples x 3 tasks)")
    print(f"Loading model: {args.model}")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1, "image": 0},
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    print("Running batch inference...")
    t0 = time.time()
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    print(f"Inference done in {time.time() - t0:.1f}s ({len(outputs)} outputs)")

    sample_outputs = {}
    for (si, task), output in zip(batch_meta, outputs):
        if si not in sample_outputs:
            sample_outputs[si] = {}
        sample_outputs[si][task] = output.outputs[0].text.strip()

    action_correct_flags = []
    action_parse_flags = []
    justification_correct_flags = []
    justification_parse_flags = []
    both_correct_flags = []
    sensibility_parse_flags = []
    sensibility_iou_scores = []
    category_correct = defaultdict(list)
    thinking_flags = []
    think_block_lengths = []
    results = []

    for si, entry in enumerate(test_data):
        if si not in sample_outputs:
            continue

        sid = entry["id"]
        meta = taxonomy.get(sid, {})
        gt_action = int(meta.get("correct", 0)) + 1
        gt_justification = gt_action
        gt_sensible_set = {
            int(idx) + 1 for idx in meta.get("sensibles", []) if isinstance(idx, int)
        }

        texts = sample_outputs[si]

        pred_action = parse_answer(texts.get("action", ""))
        action_parseable = pred_action is not None
        action_correct = pred_action == gt_action
        action_parse_flags.append(action_parseable)
        action_correct_flags.append(action_correct)

        pred_just = parse_answer(texts.get("justification", ""))
        just_parseable = pred_just is not None
        just_correct = pred_just == gt_justification
        justification_parse_flags.append(just_parseable)
        justification_correct_flags.append(just_correct)

        pred_sensible = parse_sensibility_answer(texts.get("sensibility", ""))
        sens_parseable = pred_sensible is not None
        pred_sensible_set = set(pred_sensible) if pred_sensible is not None else set()
        sens_iou = iou(pred_sensible_set, gt_sensible_set) if sens_parseable else 0.0
        sensibility_parse_flags.append(sens_parseable)
        sensibility_iou_scores.append(sens_iou)

        both_correct = action_correct and just_correct
        both_correct_flags.append(both_correct)

        sample_has_think = False
        for task_key in ("action", "justification", "sensibility"):
            raw = texts.get(task_key, "")
            if has_thinking(raw):
                sample_has_think = True
                think_block_lengths.append(len(extract_think_block(raw)))
        thinking_flags.append(sample_has_think)

        correct_idx = meta.get("correct", -1)
        categories = [c for c in meta.get("taxonomy", {}).get(str(correct_idx), []) if c]
        for cat in categories:
            category_correct[cat].append(action_correct)

        results.append(
            {
                "id": sid,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "pred_justification": pred_just,
                "pred_sensible": sorted(pred_sensible_set),
                "action_correct": action_correct,
                "justification_correct": just_correct,
                "both_correct": both_correct,
                "sensibility_iou": sens_iou,
                "categories": categories,
                "action_text": texts.get("action", ""),
                "justification_text": texts.get("justification", ""),
                "sensibility_text": texts.get("sensibility", ""),
                "action_has_thinking": has_thinking(texts.get("action", "")),
                "justification_has_thinking": has_thinking(texts.get("justification", "")),
                "sensibility_has_thinking": has_thinking(texts.get("sensibility", "")),
            }
        )

    n = len(results)
    act_acc, act_ci_lo, act_ci_hi = bootstrap_accuracy_ci(action_correct_flags)
    just_acc, just_ci_lo, just_ci_hi = bootstrap_accuracy_ci(justification_correct_flags)
    both_acc, both_ci_lo, both_ci_hi = bootstrap_accuracy_ci(both_correct_flags)
    siou, siou_ci_lo, siou_ci_hi = bootstrap_mean_ci(sensibility_iou_scores)

    per_category = {}
    for cat, flags in sorted(category_correct.items()):
        cat_acc, cat_lo, cat_hi = bootstrap_accuracy_ci(flags)
        per_category[cat] = {
            "accuracy": cat_acc,
            "ci_lower": cat_lo,
            "ci_upper": cat_hi,
            "n_samples": len(flags),
            "n_correct": sum(flags),
        }

    think_word_lengths = []
    for result in results:
        for key in ("action_text", "justification_text", "sensibility_text"):
            block = extract_think_block(result[key])
            if block:
                think_word_lengths.append(len(block.split()))

    summary = {
        "model": args.model,
        "test_path": args.test_path,
        "video_base": args.video_base,
        "nframes": args.nframes,
        "max_new_tokens": args.max_new_tokens,
        "both_accuracy": both_acc,
        "both_ci_lower_95": both_ci_lo,
        "both_ci_upper_95": both_ci_hi,
        "action_accuracy": act_acc,
        "action_ci_lower_95": act_ci_lo,
        "action_ci_upper_95": act_ci_hi,
        "ci_lower_95": act_ci_lo,
        "ci_upper_95": act_ci_hi,
        "justification_accuracy": just_acc,
        "justification_ci_lower_95": just_ci_lo,
        "justification_ci_upper_95": just_ci_hi,
        "sensibility_iou": siou,
        "sensibility_ci_lower_95": siou_ci_lo,
        "sensibility_ci_upper_95": siou_ci_hi,
        "action_parseable_rate": sum(action_parse_flags) / n if n else 0,
        "justification_parseable_rate": sum(justification_parse_flags) / n if n else 0,
        "sensibility_parseable_rate": sum(sensibility_parse_flags) / n if n else 0,
        "parseable_rate": sum(
            1
            for a, j, s in zip(
                action_parse_flags, justification_parse_flags, sensibility_parse_flags
            )
            if a and j and s
        )
        / n
        if n
        else 0,
        "total_samples": n,
        "total_action_correct": sum(action_correct_flags),
        "total_justification_correct": sum(justification_correct_flags),
        "total_both_correct": sum(both_correct_flags),
        "total_correct": sum(action_correct_flags),
        "total_action_parseable": sum(action_parse_flags),
        "total_justification_parseable": sum(justification_parse_flags),
        "total_sensibility_parseable": sum(sensibility_parse_flags),
        "total_parseable": sum(
            1
            for a, j, s in zip(
                action_parse_flags, justification_parse_flags, sensibility_parse_flags
            )
            if a and j and s
        ),
        "enable_thinking": args.enable_thinking,
        "thinking_rate": sum(thinking_flags) / n if n else 0,
        "n_think_blocks": len(think_block_lengths),
        "avg_think_block_chars": float(np.mean(think_block_lengths)) if think_block_lengths else 0,
        "avg_think_block_words": float(np.mean(think_word_lengths)) if think_word_lengths else 0,
        "median_think_block_chars": float(np.median(think_block_lengths)) if think_block_lengths else 0,
        "per_category": per_category,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "details.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Action Accuracy:        {act_acc:.1%} [{act_ci_lo:.1%}, {act_ci_hi:.1%}]")
    print(f"Justification Accuracy: {just_acc:.1%} [{just_ci_lo:.1%}, {just_ci_hi:.1%}]")
    print(f"Both Accuracy:          {both_acc:.1%} [{both_ci_lo:.1%}, {both_ci_hi:.1%}]")
    print(f"Sensibility IoU:        {siou:.4f} [{siou_ci_lo:.4f}, {siou_ci_hi:.4f}]")
    if args.enable_thinking:
        print(f"Thinking Rate:          {summary['thinking_rate']:.1%}")
        print(f"Think Blocks:           {summary['n_think_blocks']} total")
        print(
            f"Avg Think Length:       {summary['avg_think_block_chars']:.0f} chars / {summary['avg_think_block_words']:.0f} words"
        )
        print(f"Median Think Length:    {summary['median_think_block_chars']:.0f} chars")
    print("\nPer-category:")
    for cat in sorted(per_category.keys()):
        cd = per_category[cat]
        print(f"  {cat:30s} {cd['accuracy']:.1%} (n={cd['n_samples']})")
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
