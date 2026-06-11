# 11 · Agent AI Engineer — Researched, Ranked Skill Profile

What does the market actually hire an **AI engineer** to know and do? This
section answers that with real research: we gathered job descriptions from ~20
employers (frontier labs, big tech, AI-platform companies, and startups),
condensed and tallied the recurring requirements, **weighted and ranked** them,
and distilled a stable, numbered **AI-engineer skill profile**.

That ranked list is the product of this section — and it **seeds
[section 12](../12.agent-ai-skills/)**, which gives each skill its own deep
explainer.

> Authored per [`.claude/skills/tutorial-architect/SKILL.md`](../.claude/skills/tutorial-architect/SKILL.md)
> §5 (research-backed deliverables): multiple primary sources, synthesize and
> **rank** rather than copy, cite everything. No verbatim/proprietary JD text is
> reproduced — requirements are extracted and paraphrased.

---

## Files in this section (read in order)

| File | What it is |
|------|-----------|
| [`01.research-method.md`](01.research-method.md) | Sources (30 job descriptions + 12 market reports, with URLs), sample size, and exactly how skills were tallied, weighted, and ranked — plus limitations and biases. |
| [`02.job-descriptions.md`](02.job-descriptions.md) | The condensed corpus: ~30 roles across ~20 employers, each a few cited lines (company · role · level · required + nice-to-have skills). |
| [`03.ranked-skills.md`](03.ranked-skills.md) | **The centerpiece.** 32 skills ranked most→least demanded, grouped into Core / Important / Differentiating tiers, each with a "why employers want it" paragraph and rough frequency. |
| [`04.skill-frequency.md`](04.skill-frequency.md) | The frequency/importance table (skill · share · tier · trend · sources) + calibration against hard market numbers. |
| [`05.ai-engineer-agent.md`](05.ai-engineer-agent.md) | A drafted "AI Engineer" agent persona / system prompt that embodies the ranked skills, referenced by number. |
| [`agent-ai-engineer.skill.md`](agent-ai-engineer.skill.md) | The clean, numbered, machine-readable skill list (`NN. name — scope`) that **drives section 12**, plus suggested section-12 filenames. |

## How to read this section

- **In a hurry?** Read [`03.ranked-skills.md`](03.ranked-skills.md) top to bottom.
  Tier 1 (#01–#10) is the non-negotiable base; Tier 2 (#11–#22) is strong demand;
  Tier 3 (#23–#32) differentiates / gates specialist roles.
- **Want to trust the numbers?** Start with
  [`01.research-method.md`](01.research-method.md) (method + limitations), then
  [`04.skill-frequency.md`](04.skill-frequency.md) (the table and calibration).
- **Want the evidence?** [`02.job-descriptions.md`](02.job-descriptions.md) is the
  cited corpus every ranking traces back to (source IDs `S#`/`R#`).
- **Building an agent?** [`05.ai-engineer-agent.md`](05.ai-engineer-agent.md) turns
  the profile into a usable persona/system prompt.

## The skill profile at a glance

32 skills, three tiers. Core: Python/SWE · ML/DL foundations · transformer
architecture · LLM APIs & prompts · RAG · agents · evaluation · PyTorch · vector
DBs · cloud. Important: MLOps/LLMOps · fine-tuning · system design · data
engineering · communication · inference optimization · product sense · NLP ·
safety/guardrails · 2nd language · K8s · observability. Differentiating: CUDA ·
distributed training · multimodal · Hugging Face · RLHF/DPO · research literacy ·
orchestration · domain expertise · classic ML · MCP/agent protocols.

Full one-liners and stable numbering:
[`agent-ai-engineer.skill.md`](agent-ai-engineer.skill.md).

## Where this connects in the repo

The ranked skills are not abstract — most map to a hands-on tutorial elsewhere in
this curriculum:

- ML/DL/transformers → [`01.ml/`](../01.ml/), [`02.dl/`](../02.dl/), [`05.transformers/`](../05.transformers/)
- RAG · agents · evaluation · serving · fine-tuning · vectors · guardrails →
  [`07.frameworks/`](../07.frameworks/)
- Agents & MCP, clean-room → [`08.claudecode/`](../08.claudecode/)
- Hugging Face ecosystem → [`09.huggingface/`](../09.huggingface/)
- GPU/CUDA · distributed · inference optimization → [`10.gpu/`](../10.gpu/)

## ➡️ Next: one explainer per skill — [section 12 · `12.agent-ai-skills/`](../12.agent-ai-skills/)

Section 12 takes each numbered skill from
[`agent-ai-engineer.skill.md`](agent-ai-engineer.skill.md) and writes a rich
explainer (`NN.<skill>.md`): what it is, why it matters, how to learn it, tools, a
worked example, interview signals, and links back into this repo.

---

*Research snapshot: June 2026. The AI-engineering job market moves fast; see the
trend column in [`04.skill-frequency.md`](04.skill-frequency.md) for direction of
change, and [`01.research-method.md`](01.research-method.md) §7 to refresh.*
