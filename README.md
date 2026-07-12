# LLM From Scratch

This repository organizes self-paced LLM course work into two parallel paths:

- `Answer/`: completed-code reference implementations.
- `Human/`: the learner workspace and project notes for implementing assignments by hand.

The main learning artifact in each Human project is `Note.md`. These notes explain the assignment from a top-down perspective, then point to the exact files where code should be written and the tests that verify each task.

## Structure

```text
Answer/
  0_torch_basics/
  1_autodiff/
  2_tensor_ops/
  3_transformer_basics/
  4_distributed_systems/
  5_scaling_laws/
  6_data_pipeline/
  7_grpo_alignment/

Human/
  0_torch_basics/
  1_autodiff/
  2_tensor_ops/
  3_transformer_basics/
  4_distributed_systems/
  5_scaling_laws/
  6_data_pipeline/
  7_grpo_alignment/
```

## How To Use

Start with the relevant `Human/.../Note.md`, especially the section titled `第二章：如何完成 Human 路径`. It tells you:

1. Which `Human/...` files to edit.
2. What each task should accomplish.
3. How to test each task.

Then implement the corresponding code under `Human/` and run the listed tests from that assignment directory.
