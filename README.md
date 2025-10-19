# AI Tools Classification & Summarization using IBM Granite

## Project Overview
This project aims to **classify and summarize generative AI tools** based on their primary modality — such as **code, image, video, audio, text, or infrastructure** — using the **IBM Granite 3.3 8B Instruct model** through the **Replicate API**.  
It automates the analysis of an AI tools dataset to produce two key outcomes:
1. **Classification Output** — A grouped list of AI tools per modality (up to 5 representative tools each).
2. **Summarization Output** — A concise analytical summary highlighting trends, dominance, and use cases among modalities.

The workflow integrates **LangChain + Replicate** for LLM orchestration and uses **Pandas** for dataset preprocessing.

---

## Raw Dataset
The raw dataset used in this analysis:
**[Generative AI Tools – Platforms 2025.csv](./data/Generative%20AI%20Tools%20-%20Platforms%202025.csv)** from https://www.kaggle.com/datasets/tarekmasryo/generative-ai-tools-and-platforms-2025
Each record includes:
- `tool_name`: Name of the AI platform/tool  
- `modality_canonical`: The modality or application type (e.g., code, text, image, video, etc.)

---

## Insights & Findings

### Classification
The model categorized AI tools into modalities such as:
- **Code** — tools like *ChatGPT*, *Replit*, *GitHub Copilot*, *Tabnine*, *Codeium*  
- **Image** — tools like *Midjourney*, *DALL·E 3*, *Stable Diffusion*, *Firefly*, *Leonardo AI*  
- **Video** — tools like *Runway ML*, *Pika Labs*, *Synthesia*, *HeyGen*, *Kaiber*  
- **Text** — tools like *Claude*, *Gemini*, *Perplexity*, *Writer*, *Copy.ai*  
- **Infra** — tools like *Replicate*, *Hugging Face*, *OctoAI*, *Modal*, *Vercel AI SDK*

### Summarization (Example Output)
> The dataset shows a growing diversity of AI modalities, with **text- and code-based models** remaining the most common due to wide developer adoption.  
> Image and video modalities are expanding rapidly, driven by content generation demand.  
> Infrastructure tools serve as essential backbones, enabling scalable model deployment and accessibility for research and production.

---

## AI Support Explanation
This project leverages **IBM Granite 3.3 (8B-Instruct)** via **Replicate API** for both classification and summarization:

| Component | Role | Technology |
|------------|------|-------------|
| **IBM Granite LLM** | Processes natural-language instructions for grouping and analysis | `ibm-granite/granite-3.3-8b-instruct` |
| **LangChain Community** | Provides standardized LLM interface for Replicate | `langchain_community.llms.Replicate` |
| **Pandas** | Handles data loading, cleaning, and formatting for prompts | `pandas` |
| **dotenv** | Manages API key securely via `.env` | `python-dotenv` |

Granite model acts as an **analytical LLM**, capable of recognizing semantic relations between AI tools and categorizing them based on descriptive metadata.  
This allows fully automated classification and insight generation without manual labeling.

---

## Requirements
- Python ≥ 3.9  
- Replicate API Token  
- Dataset CSV file (provided in `data/` folder) from www.kaggle.com

Install dependencies:
```bash

pip install -r requirements.txt

