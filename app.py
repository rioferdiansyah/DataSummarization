import os
import pandas as pd
from langchain_community.llms import Replicate
from dotenv import load_dotenv

# STEP 1 — LOAD ENVIRONMENT VARIABLE
load_dotenv()
api_token = os.getenv("REPLICATE_API_TOKEN")

if not api_token:
    raise ValueError("Token Replicate tidak ditemukan. Tambahkan ke file .env")

os.environ["REPLICATE_API_TOKEN"] = api_token

# STEP 2 — LOAD DATASET
csv_path = "Generative AI Tools - Platforms 2025.csv"
df = pd.read_csv(csv_path)

print("Dataset loaded successfully. Preview:")
print(df.head())

# Pastikan kolom yang diperlukan ada
required_columns = ["tool_name", "modality_canonical"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.")

# STEP 3 — INISIALISASI MODEL IBM GRANITE
model = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    model_kwargs={
        "max_tokens": 512,
        "top_p": 0.9,
        "temperature": 0.6,
    },
)

# STEP 4 — SIAPKAN DATA UNTUK PROMPT
subset = df[["tool_name", "modality_canonical"]].dropna()
table_text = "\n".join([f"{row.tool_name} | {row.modality_canonical}" for _, row in subset.iterrows()])

# STEP 5 — PROMPT UNTUK CLASSIFICATION
classification_prompt = """
You are an AI assistant that analyzes a dataset of generative AI tools.

Task:
Group the AI tools listed in the table below based on their modality type (column: modality_canonical).

For each modality (like code, image, video, infra, audio, text, etc.), list:
- The modality name as a numbered list (1., 2., 3., ...)
- Up to 5 representative tool names for that modality, separated by commas.

Do not use tables.
Do not explain or summarize — just output the clean grouped list in this exact format:

1. code: Tool1, Tool2, Tool3, Tool4, Tool5
2. image: Tool1, Tool2, Tool3, Tool4, Tool5
3. infra: Tool1, Tool2, Tool3, Tool4, Tool5
4. video: Tool1, Tool2, Tool3, Tool4, Tool5
5. text: Tool1, Tool2, Tool3, Tool4, Tool5

If a modality has fewer than 5 tools, include only those available.

Now process the dataset below.

Input columns:
- tool_name
- modality_canonical
"""

# STEP 6 — JALANKAN CLASSIFICATION
print("\nRunning classification with IBM Granite...\n")
classification_output = model.invoke(classification_prompt + "\n\nDataset:\n" + table_text)

print("=== CLASSIFICATION RESULT ===\n")
print(classification_output)

# STEP 7 — PROMPT UNTUK SUMMARIZATION
summarization_prompt = f"""
You are an AI analyst.

Based on the grouped list of AI tools below, write a short and clear summary that highlights:
- Which modalities (code, image, video, infra, etc.) are most common.
- The general trend or use cases across modalities.
- Any interesting observations (e.g., dominance of certain tool types, growth areas).

Keep the tone analytical but concise.
Output should be 3–5 short paragraphs only.

Grouped List:
{classification_output}
"""

# STEP 8 — JALANKAN SUMMARIZATION
print("\nGenerating summarization insight...\n")
summary_output = model.invoke(summarization_prompt)

print("=== SUMMARIZATION RESULT ===\n")
print(summary_output)

print("\nAll steps completed successfully.")