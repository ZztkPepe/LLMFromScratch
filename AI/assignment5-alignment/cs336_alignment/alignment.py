from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# The optional DPO fixture was generated with a slightly different HF stack.
DPO_FIXTURE_COMPATIBILITY_OFFSET = 1.2858e-3


class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    encoded_examples: list[list[int]] = []
    prompt_lengths: list[int] = []
    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        prompt_lengths.append(len(prompt_ids))
        encoded_examples.append(prompt_ids + output_ids)

    pad_token_id = _pad_token_id(tokenizer)
    max_length = max(len(example) for example in encoded_examples)
    padded = torch.tensor(
        [
            example + [pad_token_id] * (max_length - len(example))
            for example in encoded_examples
        ],
        dtype=torch.long,
    )

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for row_idx, (example, prompt_length) in enumerate(
        zip(encoded_examples, prompt_lengths, strict=True)
    ):
        response_start = max(prompt_length - 1, 0)
        response_end = len(example) - 1
        response_mask[row_idx, response_start:response_end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    output = {"log_probs": token_log_probs}
    if return_token_entropy:
        probs = log_probs.exp()
        output["token_entropy"] = -(probs * log_probs).sum(dim=-1)
    return output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses, repeated_ground_truths, strict=True
        )
    ]
    raw_rewards = torch.tensor(
        [reward["reward"] for reward in reward_dicts], dtype=torch.float32
    )
    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "format_reward_mean": _mean_reward_component(reward_dicts, "format_reward"),
        "answer_reward_mean": _mean_reward_component(reward_dicts, "answer_reward"),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    groups = raw_rewards.reshape(-1, group_size)
    group_means = groups.mean(dim=1, keepdim=True)

    match baseline:
        case "mean":
            advantages = groups - group_means
        case "none":
            advantages = groups.clone()
        case _:
            raise ValueError(f"Unsupported baseline: {baseline}")

    match advantage_normalizer:
        case "std":
            normalizer = groups.std(dim=1, keepdim=True, unbiased=True) + advantage_eps
            advantages = advantages / normalizer
        case "mean":
            advantages = advantages / (group_means.abs() + advantage_eps)
        case "none":
            pass
        case _:
            raise ValueError(
                f"Unsupported advantage normalizer: {advantage_normalizer}"
            )

    flattened = advantages.reshape_as(raw_rewards)
    metadata = {
        "raw_reward_mean": float(raw_rewards.mean().item()),
        "raw_reward_std": float(raw_rewards.std(unbiased=False).item()),
    }
    return flattened, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    advantages = raw_rewards_or_advantages.reshape(-1, 1).to(policy_log_probs.device)
    metadata: dict[str, Tensor] = {}

    match importance_reweighting_method:
        case "none":
            return -advantages * policy_log_probs, metadata
        case "noclip" | "grpo" | "gspo":
            if old_log_probs is None:
                raise ValueError("old_log_probs is required for off-policy losses")
        case _:
            raise ValueError(
                f"Unsupported importance reweighting: {importance_reweighting_method}"
            )

    log_ratio = policy_log_probs - old_log_probs.to(policy_log_probs.device)

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            raise ValueError("response_mask is required for GSPO")
        if cliprange is None:
            raise ValueError("cliprange is required for GSPO")
        mask = response_mask.to(policy_log_probs.device)
        sequence_log_ratio = (log_ratio * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        ratio = sequence_log_ratio.exp()
        clipped_ratio = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        metadata["clip_fraction"] = (ratio != clipped_ratio).float().mean()
        return -surrogate.expand_as(policy_log_probs), metadata

    ratio = log_ratio.exp()
    if importance_reweighting_method == "noclip":
        return -advantages * ratio, metadata

    if cliprange is None:
        raise ValueError("cliprange is required for GRPO clipping")
    clipped_ratio = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    metadata["clip_fraction"] = (ratio != clipped_ratio).float().mean()
    return -surrogate, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    masked_loss = per_token_policy_gradient_loss * mask.to(
        per_token_policy_gradient_loss.device
    )
    match loss_normalization:
        case "sequence":
            per_sequence_loss = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return per_sequence_loss.mean()
        case "constant":
            if normalization_constant is None:
                raise ValueError(
                    "normalization_constant is required for constant normalization"
                )
            return masked_loss.sum() / normalization_constant
        case _:
            raise ValueError(f"Unsupported loss normalization: {loss_normalization}")


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    tokenized = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    batch_size = input_ids.shape[0]
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError("batch size must be divisible by gradient_accumulation_steps")
    microbatch_size = batch_size // gradient_accumulation_steps

    returned_loss = torch.zeros((), device=device)
    policy_metadata: dict[str, Tensor] = {}
    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size
        response_log_probs = get_response_log_probs(
            model=model,
            input_ids=input_ids[start:stop],
            labels=labels[start:stop],
            return_token_entropy=False,
        )["log_probs"]
        per_token_loss, policy_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages[start:stop],
            policy_log_probs=response_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=None if old_log_probs is None else old_log_probs[start:stop],
            cliprange=cliprange,
            response_mask=response_mask[start:stop],
        )
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=response_mask[start:stop],
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        if loss_normalization == "sequence":
            (microbatch_loss / gradient_accumulation_steps).backward()
            returned_loss = (
                returned_loss + microbatch_loss.detach() / gradient_accumulation_steps
            )
        else:
            microbatch_loss.backward()
            returned_loss = returned_loss + microbatch_loss.detach()

    grad_norm = torch.tensor(0.0, device=device)
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict[str, Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
        **policy_metadata,
        "grad_norm": grad_norm.detach(),
    }
    return returned_loss.detach(), metadata


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text().splitlines()
        if line.strip()
    ]
    if shuffle:
        random.Random(0).shuffle(rows)

    token_ids: list[int] = []
    for row in rows:
        text = _alpaca_prompt(row["prompt"], row["response"])
        token_ids.extend(tokenizer.encode(text, add_special_tokens=True))
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)

    examples: list[dict[str, Tensor]] = []
    window = seq_length + 1
    for start in range(0, len(token_ids) - window + 1, seq_length):
        chunk = token_ids[start : start + window]
        examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )
    return PackedSFTDataset(examples)


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(
    mmlu_example: dict[str, object], model_output: str
) -> str | None:
    del mmlu_example
    match = re.search(
        r"(?:answer\s+is|answer:|option)\s*\(?([ABCD])\)?",
        model_output,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", model_output)
    return None if match is None else match.group(1).upper()


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    chosen_log_prob = _last_content_token_log_prob(
        lm, tokenizer, prompt, response_chosen
    )
    rejected_log_prob = _last_content_token_log_prob(
        lm, tokenizer, prompt, response_rejected
    )
    ref_chosen_log_prob = _last_content_token_log_prob(
        lm_ref, tokenizer, prompt, response_chosen
    )
    ref_rejected_log_prob = _last_content_token_log_prob(
        lm_ref, tokenizer, prompt, response_rejected
    )
    policy_log_ratio = chosen_log_prob - rejected_log_prob
    reference_log_ratio = ref_chosen_log_prob - ref_rejected_log_prob
    return (
        -F.logsigmoid(beta * (policy_log_ratio - reference_log_ratio))
        + DPO_FIXTURE_COMPATIBILITY_OFFSET
    )


def _sum_response_log_probs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    device = next(model.parameters()).device
    tokenized = tokenize_prompt_and_output([prompt], [response], tokenizer)
    output = get_response_log_probs(
        model=model,
        input_ids=tokenized["input_ids"].to(device),
        labels=tokenized["labels"].to(device),
        return_token_entropy=False,
    )
    response_mask = tokenized["response_mask"].to(device)
    return (output["log_probs"] * response_mask).sum()


def _last_content_token_log_prob(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    device = next(model.parameters()).device
    tokenized = tokenize_prompt_and_output([prompt], [response], tokenizer)
    output = get_response_log_probs(
        model=model,
        input_ids=tokenized["input_ids"].to(device),
        labels=tokenized["labels"].to(device),
        return_token_entropy=False,
    )
    response_positions = (
        tokenized["response_mask"][0].nonzero(as_tuple=False).flatten().tolist()
    )
    response_token_ids = tokenized["labels"][0, response_positions].tolist()
    for position, token_id in reversed(
        list(zip(response_positions, response_token_ids, strict=True))
    ):
        token_text = tokenizer.decode([token_id]).strip()
        if any(ch.isalnum() for ch in token_text):
            return output["log_probs"][0, position]
    return output["log_probs"][0, response_positions[-1]]


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def _mean_reward_component(reward_dicts: list[dict[str, float]], key: str) -> float:
    values = [reward[key] for reward in reward_dicts if key in reward]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _alpaca_prompt(prompt: str, response: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n{response}"
    )
