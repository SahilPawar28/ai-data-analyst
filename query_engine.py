from dotenv import load_dotenv
import ast
import os
import numpy as np
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")


def build_query_prompt(question, df):
    col_info = []
    for col in df.columns:
        col_info.append(f"- {col} ({df[col].dtype})")

    columns_text = "\n".join(col_info)
    sample = df.head(3).to_string(index=False)

    prompt = f"""You are a Python Pandas expert.
A user has a DataFrame called `df` with the following columns:

{columns_text}

Sample data:
{sample}

The user asks: "{question}"

Write a single Python expression using `df` that answers this question.
Rules:
- Return ONLY the Python expression, nothing else
- No imports, no assignments, no print statements
- Must be a single line expression
- Use only Pandas operations
- The result should be a value, Series, or small DataFrame

Expression:"""

    return prompt


def _extract_expression(text):
    """Pull a single pandas expression out of a model response that may
    contain markdown fences and/or chain-of-thought reasoning around it."""
    text = text.strip()

    if "```" in text:
        fenced = text.split("```")[1]
        fenced = fenced[len("python"):] if fenced.startswith("python") else fenced
        fenced = fenced.strip().strip("\n")
        if fenced:
            return fenced

    # No usable fence: reasoning models often think out loud before (or
    # after, if truncated) the real expression. Scan every line, last
    # first, and take the first one that's syntactically valid Python
    # containing `df` — that's very unlikely to be a stray reasoning
    # sentence.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if "df" not in line:
            continue
        try:
            ast.parse(line, mode="eval")
            return line
        except SyntaxError:
            continue

    return lines[-1] if lines else text


def execute_query(question, df):
    prompt = build_query_prompt(question, df)
    system_msg = "You are a Pandas expert. Return only a single Python expression, no explanation."

    generated_code = None
    last_error = None

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        if not response.choices:
            upstream_msg = getattr(response, "error", None)
            last_error = upstream_msg.get("message") if isinstance(upstream_msg, dict) else "model provider error"
        else:
            text = response.choices[0].message.content
            if text:
                generated_code = _extract_expression(text)
            else:
                last_error = "AI returned an empty response."
    except Exception as e:
        last_error = str(e)

    if generated_code is None:
        return f"❌ Could not process query: {last_error}", None

    try:
        result = eval(generated_code, {"__builtins__": {}, "pd": pd, "np": np}, {"df": df})

        if isinstance(result, pd.DataFrame):
            # Keep the index when it carries information (e.g. set_index('year'));
            # only drop it when it's a meaningless default RangeIndex.
            keep_index = not isinstance(result.index, pd.RangeIndex)
            raw_result = result.head(10).to_string(index=keep_index)
        elif isinstance(result, pd.Series):
            raw_result = result.head(10).to_string()
        else:
            raw_result = str(result)

        # Step 2: Use AI to format the answer in plain text
        format_prompt = f"""You are a helpful data analyst.
The user asked: "{question}"
The data query returned this raw result:
{raw_result}

Provide a clear, concise, natural-language answer to the user's question based on this result. Do not use tables or code blocks. Just plain text."""

        final_answer = raw_result  # Fallback
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": format_prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            text = response.choices[0].message.content
            if text:
                final_answer = text.strip()
        except Exception:
            pass

        return final_answer, generated_code

    except Exception as e:
        return f"❌ Could not execute generated code: {str(e)}", generated_code
