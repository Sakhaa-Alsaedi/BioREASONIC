# BioREASONIC Executed Experiment (Notebooks)

**Paper**: BioREASONIC: A Causal-Oriented GraphRAG System for Multi-Aware Biomedical Reasoning  
**Authors**: Sakhaa Alsaedi, Mohammed Saif, Takashi Gojobori, Xin Gao  
**Submitted to**: ISMB 2026

---

## Reproducibility Notice

All experiments require **LLM API keys**. Due to API availability constraints, we provide:
- Jupyter notebooks to reproduce results
- Pre-computed outputs for reference

**To run the code yourself, provide your own API keys.**

---

## API Keys Required

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| OpenAI | GPT-4.1, GPT-4.1-mini, GPT-4o, GPT-4o-mini | `OPENAI_API_KEY` |
| Anthropic | Claude-3-Haiku | `ANTHROPIC_API_KEY` |
| DeepSeek | DeepSeek-V3 | `DEEPSEEK_API_KEY` |
| Together AI | LLaMA-3.1-8B, Qwen-2.5-7B | `TOGETHER_API_KEY` |

---

## Notebooks and Experiments

| Notebook | Manuscript Table/Figure | Task |
|----------|------------------------|------|
| `MedQA_biomedical_llm_complete_n_Samples.ipynb' | - | Testing the impact of batch size [50, 100, 200, 500, 1000] on LLM performance|
| `KGQA_with_result.ipynb` , `Prompting_Strategy_Evaluation.ipynb' | Table 2, Figure 5 |Prompting strategy comparison |
| `ELV_with_result.ipynb` | Table 3, Table 5 | Expert-Level Verification (ELV) |
| `KGQA_with_result.ipynb` | Table 4 | KGQA ablation study |

---

## Estimated Token Usage (200 samples per benchmark)

| Model | Tokens/Sample | Total Tokens | Est. Cost |
|-------|---------------|--------------|-----------|
| GPT-4.1 | ~1,000 | ~200K | ~$6.00 |
| GPT-4.1-mini | ~900 | ~180K | ~$0.36 |
| GPT-4o | ~900 | ~180K | ~$4.50 |
| GPT-4o-mini | ~880 | ~176K | ~$0.26 |
| Claude-3-Haiku | ~1,120 | ~224K | ~$0.56 |
| DeepSeek-V3 | ~930 | ~186K | ~$0.26 |
| LLaMA-3.1-8B | ~1,010 | ~202K | ~$0.04 |
| Qwen-2.5-7B | ~1,090 | ~218K | ~$0.04 |

**Note**: Multi-Agent baseline uses ~4× more tokens (see Table 5).

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:
```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
TOGETHER_API_KEY=your_key
```

