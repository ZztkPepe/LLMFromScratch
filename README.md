# LLM From Scratch

This repository organizes self-paced LLM course work into two parallel paths:

- `AI/`: reference-oriented course notes and completed-code context.
- `Human/`: the learner workspace for implementing assignments by hand.

The main learning artifact in each AI module or assignment is `Note.md`. These notes are written to explain the assignment from a top-down perspective, then guide the Human path by pointing to the exact files where code should be written and the tests that verify each task.

## Structure

```text
AI/
  Module-0/
  Module-1/
  Module-2/
  assignment1-basics/
  assignment2-systems/
  assignment3-scaling/
  assignment4-data/
  assignment5-alignment/

Human/
  Module-0/
  Module-1/
  Module-2/
  assignment1-basics/
  assignment2-systems/
  assignment3-scaling/
  assignment4-data/
  assignment5-alignment/
```

## How To Use

Start with the relevant `AI/.../Note.md`, especially the section titled `第二章：如何完成 Human 路径`. It tells you:

1. Which `Human/...` files to edit.
2. What each task should accomplish.
3. How to test each task.

Then implement the corresponding code under `Human/` and run the listed tests from that assignment directory.
