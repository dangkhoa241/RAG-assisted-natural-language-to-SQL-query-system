---
title: NL to SQL Data Assistant
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Plain-English questions to SQL, with business-glossary RAG
---

# NL → SQL Data Assistant

Ask questions about a CSV in plain English and get the SQL, a table and a chart. Try the three sample
datasets (healthcare, retail, SaaS) or upload your own CSV (up to 10 MB).

- A fine-tuned BERT classifier predicts the question's intent.
- **gpt-oss-120b on Groq** writes the SQL from the dataset's schema. If its quota runs out, gpt-oss-20b
  takes over, then a rule-based generator.
- When a question uses a company-specific business term ("ARR", "churned", "senior patient"), only that
  term's glossary definition is added to the prompt.
- Every query runs read-only, time- and row-capped. The "How it works" panel shows each step.

Source code, benchmarks and results: [GitHub](https://github.com/dangkhoa241/RAG-assisted-natural-language-to-SQL-query-system).

This Space is deployed with `scripts/deploy_space.py` from that repo. A free Space sleeps when idle, so the
first visit may take about a minute to wake up.
