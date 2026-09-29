# Customer Intelligence Platform

Combines **behavioral** e-commerce signals (orders, browsing, funnel) with **conversational** signals
(support chats: intent, sentiment) into a unified customer profile, then produces
**evidence-grounded recommendations** with retrieval (pgvector) and a local LLM (Ollama).

> Status: milestone 1–2 of the MVP: reproducible synthetic data + sanity checks.

## Why synthetic data, and how it avoids being meaningless

Public datasets don't link purchases and support conversations for the same customers. So the
world is simulated, but from a **latent-state model**, not independent random tables:

- Each customer has hidden parameters: purchase rate λ ~ Gamma, dropout probability p ~ Beta,
  category affinities ~ Dirichlet, price preference, delivery-issue exposure.
- Purchases follow a **BG/NBD-style** process: Poisson arrivals while active, dropout with
  probability p after each purchase. A delivery problem raises p (the complaint → churn link).
- Browsing sessions, abandoned carts and support contacts are all sampled from the same hidden state.
- Conversation **text is real** (Bitext customer-support dataset, sampled by intent), not LLM-written,
  to avoid a circular evaluation.

The pipeline only reads `data/raw/`. Evaluation code alone reads `data/ground_truth/`
(true activity status, conversation labels, oracle best action). Because the true state is
known, we can measure how well the system recovers it. The best achievable score is below 100%,
as it would be in reality.

### Known limitations (stated, not hidden)
- The behavior ↔ conversation linkage is synthetic; its strength is a config choice.
- Bitext utterances are chosen by intent only, so their tone isn't driven by latent frustration.
  Sentiment is therefore evaluated on a hand-labelled sample.
- The "best action" oracle is a documented rule set (`src/cip/data/oracle.py`), not a causal truth.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[text,dev]"
python scripts/generate_data.py          # downloads Bitext once (~27k utterances)
python scripts/sanity_report.py          # checks the intended structure exists
pytest
```
Offline smoke test: `python scripts/generate_data.py --text-source templates` (do not report metrics on it).

## Layout
```
configs/generator.yaml      every modelling assumption, seeded
src/cip/data/               generator, catalog, conversations, oracle
scripts/                    CLI entry points
tests/                      invariants: reproducibility, no leakage, no time travel
docker-compose.yml          Postgres + pgvector (used from milestone 6)
```

## Roadmap
- [x] M1–2 Reproducible synthetic world + sanity checks
- [ ] M3 RFM, funnel, personas (rule-based, validated against true activity status)
- [ ] M4 Intent (TF-IDF baseline vs SetFit learning curve), sentiment (pretrained RoBERTa)
- [ ] M5 Unified customer profile
- [ ] M6 Knowledge base + pgvector retrieval + Recall@k / MRR
- [ ] M7 Ollama recommender with structured output + citation validation
- [ ] M8 Ablation evaluation, Streamlit demo → v0.1
