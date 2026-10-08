from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")


def build_summary_prompt(summary, statistics, distribution):
    prompt = "You are an expert data analyst. Based on the following dataset summary, generate 5 clear, actionable business insights.\n\n"

    prompt += "Dataset Overview:\n"
    prompt += f"- Rows: {summary['rows']}\n"
    prompt += f"- Columns: {summary['columns']}\n"
    prompt += f"- Column Names: {', '.join(summary['column_names'])}\n"
    prompt += f"- Total Missing Values: {summary['total_missing']}\n\n"

    if statistics:
        prompt += "Descriptive Statistics:\n"
        for col, stats in statistics.items():
            prompt += f"- {col}: mean={stats['mean']}, median={stats['median']}, min={stats['min']}, max={stats['max']}, std={stats['std_dev']}\n"
        prompt += "\n"

    if distribution:
        prompt += "Categorical Column Distributions:\n"
        for col, data in distribution.items():
            top_items = list(data["percentages"].items())[:3]
            for item, pct in top_items:
                prompt += f"- {col}: {item} → {pct}%\n"
        prompt += "\n"

    prompt += "Generate exactly 5 numbered business insights. Be specific, concise, and actionable."
    return prompt


def generate_insights(summary, statistics, distribution):
    prompt = build_summary_prompt(summary, statistics, distribution)
    system_msg = "You are a senior data analyst. Provide exactly 5 numbered, detailed business insights based on the summary. Do not use markdown headers, just simple text."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        text = response.choices[0].message.content
        if text:
            return text.strip(), None
        return None, "❌ AI Error: Empty response from model."

    except Exception as e:
        return None, f"❌ AI Error: {str(e)}"
