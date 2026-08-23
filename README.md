# Stanza AI — Intent Classification Model Fine-Tuning & Comparison

## What This Project Is

This repository documents an experiment to find the best open-source LLM for building the **Intent Classification / Router Agent** inside **Stanza AI** — a multi-agent LLM layer designed for a hospitality operations platform (StanzaFlex).

In Stanza AI's architecture, every guest or staff request first passes through an **Orchestrator Agent**, whose job is to read the incoming query and decide which specialist agent should handle it (Reservations, Housekeeping, Front Desk, F&B, Revenue & Pricing, Concierge, or Reporting & Analytics). This project focuses specifically on that first, critical step: **can a small, fine-tuned open-source model reliably classify a guest query into the correct agent and intent, cheaply and quickly?**

## Why This Was Made

Relying on a large closed-source API (like GPT) for every single incoming query is expensive and adds latency, especially for a task as narrow and repetitive as routing. The hypothesis behind this project is that a **small, fine-tuned open-source model can match or approach that routing accuracy at a fraction of the cost**, and can run in-house instead of depending on an external API for every request.

To test this, the goal was to:
1. Build a clean, labeled dataset of realistic hotel guest/staff queries.
2. Fine-tune multiple candidate open-source LLMs on that dataset using parameter-efficient fine-tuning (LoRA).
3. Evaluate each model on unseen test queries.
4. Pick a winner to scale up to the full production dataset.

## How This Is Beneficial

- **Cost reduction** — a small fine-tuned model doing routing removes the need to call an expensive general-purpose API for every incoming message.
- **Lower latency** — routing is the first step in every request; a fast, local model keeps response times low.
- **Data ownership** — fine-tuning in-house means guest query patterns and routing logic stay within the organization's own model, rather than being sent to a third party for every request.
- **A repeatable evaluation process** — the same dataset, training method, and evaluation script can be reused to test future candidate models as new open-source LLMs are released.

## Dataset

- **700 synthetic, English-only records**, evenly split across the 7 specialist agents (100 records each).
- Each record follows this schema:

| Column | Description |
|---|---|
| `record_id` | Unique identifier |
| `agent_label` | The specialist agent responsible for the query |
| `intent_name` | The specific intent label (snake_case) |
| `query` | The guest/staff message in plain English |
| `entities` | Key information extracted from the query (JSON) |
| `language` | Language code (`en`) |
| `split` | `train` / `validation` / `test` |

- **Split:** 560 train / 70 validation / 70 test.
- Intent naming and agent groupings were standardized against a shared reference schema so multiple contributors could build data for different agents without naming collisions.
- All queries and entities are synthetically generated (no real guest data), covering realistic hospitality scenarios (bookings, complaints, room service, housekeeping requests, pricing inquiries, etc.).

## Models Compared

Three open-source, commercially-licensed instruction-tuned LLMs were shortlisted and fine-tuned under identical conditions:

| Model | Size | License |
|---|---|---|
| Qwen2.5-7B-Instruct | 7B | Apache 2.0 |
| Mistral-7B-Instruct-v0.3 | 7B | Apache 2.0 |
| Phi-4-mini-instruct | 3.8B | MIT |

**Fine-tuning method (identical for all three):**
- QLoRA (4-bit quantization + LoRA adapters)
- LoRA rank = 16, alpha = 32, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- 3 epochs, batch size 2 (gradient accumulation 4), learning rate 2e-4
- Trained on Google Colab (T4 GPU)
- Task framed as instruction-following JSON generation: given a query, the model outputs `{"agent_label": ..., "intent_name": ...}`

## Results

Evaluated on the 70-record held-out test set, comparing the model's predicted `agent_label` and `intent_name` against the ground truth:

| Model | Overall Accuracy (agent + intent both correct) | Agent-only Accuracy | Intent-only Accuracy |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** 🏆 | **88.57%** | **97.14%** | **88.57%** |
| Mistral-7B-Instruct-v0.3 | 84.29% | 91.43% | 85.71% |
| Phi-4-mini-instruct | 78.57% | 92.86% | 81.43% |

### Comparison / Observations

- **Qwen2.5-7B-Instruct won on all three metrics**, making it the clear choice for scaling up.
- **Agent-level routing was consistently stronger than intent-level classification** across all three models — meaning all models found it easier to identify *which department* a query belongs to than to pin down the *exact fine-grained intent* within that department.
- **Phi-4-mini** had a decent agent-routing score (92.86%) despite being roughly half the size of the other two models, but its intent-level accuracy dropped the most — suggesting smaller models lose precision on fine-grained distinctions faster than they lose it on coarse routing.
- **Mistral-7B** performed reasonably but trailed Qwen on every metric under identical training conditions.

### Conclusion

**Qwen2.5-7B-Instruct is selected as the winning model** and will be fine-tuned next on the full 5,000-record production dataset.

## Project Structure

```
stanza-ai-fine-tuning-comparison/
├── README.md
├── data/
│   └── Stanza_AI_Intent_Classification_Final_700.csv   # training dataset (700 records)
├── notebooks/
│   ├── 01_Qwen_Training.ipynb                          # Qwen fine-tuning + evaluation
│   ├── 02_Mistral_Training.ipynb                       # Mistral fine-tuning + evaluation
│   └── 03_Phi_Training.ipynb                           # Phi fine-tuning + evaluation
└── results/
    ├── qwen_evaluation_results.csv                     # per-query predictions vs ground truth
    ├── mistral_evaluation_results.csv
    └── phi_evaluation_results.csv
```

## Next Steps

- Fine-tune Qwen2.5-7B-Instruct on the full 5,000-record production dataset.
- Extend evaluation with additional metrics (per-agent precision/recall, confusion matrix) once the larger dataset is ready.
- Re-introduce multilingual (Hindi/Hinglish) queries for a future evaluation round, since the current dataset is English-only.
