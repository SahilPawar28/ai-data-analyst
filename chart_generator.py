from dotenv import load_dotenv
import ast
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# Modules the generated expression is allowed to use. No other names resolve,
# and __builtins__ is explicitly empty below — this is what actually prevents
# a generated expression from reaching `open`, `__import__`, `os`, etc.
SAFE_GLOBALS = {
    "__builtins__": {},
    "pd": pd,
    "px": px,
    "go": go,
    "np": np,
}


def build_chart_prompt(request, df):
    col_info = "\n".join(f"- {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(3).to_string(index=False)

    return f"""You are a Python data visualization expert using Plotly.
A user has a DataFrame called `df` with these columns:

{col_info}

Sample data:
{sample}

The user wants this chart: "{request}"

Write a single Python expression using `df`, `px` (plotly.express), `go` (plotly.graph_objects),
`pd`, and `np` that builds a Plotly Figure and answers the request.

Rules:
- Return ONLY the Python expression, nothing else
- No imports, no assignments, no print statements, no multi-line blocks
- Must be a single expression that evaluates to a plotly Figure (e.g. a px.bar(...) call,
  or a go.Figure(...) call)
- If the request implies aggregation (e.g. "average X by Y"), aggregate the data first
  as part of the same expression (e.g. using a chained .groupby(...).mean().reset_index()
  inline before passing it to px)
- Use template="plotly_dark" as a keyword argument
- If a requested grouping column has very high cardinality (many more than ~15 unique
  values) and the chart type is a pie or bar chart, aggregate to the top 10 by value first

Expression:"""


def _extract_expression(text):
    """Pull a single Python expression out of a model response that may
    contain markdown fences and/or chain-of-thought reasoning around it."""
    text = text.strip()

    if "```" in text:
        fenced = text.split("```")[1]
        fenced = fenced[len("python"):] if fenced.startswith("python") else fenced
        fenced = fenced.strip().strip("\n")
        if fenced:
            return fenced

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if "px." not in line and "go." not in line:
            continue
        try:
            ast.parse(line, mode="eval")
            return line
        except SyntaxError:
            continue

    return lines[-1] if lines else text


def generate_chart_from_prompt(request, df):
    """
    Turns a natural-language chart request into a Plotly figure.
    Returns (figure, generated_code, error) — exactly one of figure/error is set.
    """
    prompt = build_chart_prompt(request, df)
    system_msg = (
        "You are a Plotly expert. Return only a single Python expression that "
        "builds a plotly Figure, no explanation."
    )

    generated_code = None
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        if not response.choices:
            upstream_msg = getattr(response, "error", None)
            detail = upstream_msg.get("message") if isinstance(upstream_msg, dict) else "no response"
            return None, None, f"❌ Model provider error, try again shortly: {detail}"

        text = response.choices[0].message.content
        if not text:
            return None, None, "❌ AI returned an empty response."
        generated_code = _extract_expression(text)
    except Exception as e:
        return None, None, f"❌ Could not reach the model: {str(e)}"

    try:
        tree = ast.parse(generated_code, mode="eval")
    except SyntaxError as e:
        return None, generated_code, f"❌ Generated code is not a valid expression: {str(e)}"

    # Reject import statements before even considering evaluation. Lambdas
    # (e.g. df.assign(x=lambda d: ...)) are common, safe pandas idioms and
    # are NOT blocked here — the empty __builtins__ in SAFE_GLOBALS is what
    # actually prevents sandbox escapes, independent of lambda syntax.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, generated_code, "❌ Generated code used a disallowed import."

    try:
        fig = eval(generated_code, SAFE_GLOBALS, {"df": df})
    except Exception as e:
        return None, generated_code, f"❌ Could not execute generated chart code: {str(e)}"

    if not isinstance(fig, (go.Figure,)):
        return None, generated_code, "❌ Generated code did not produce a chart."

    return fig, generated_code, None


def export_chart_image(fig, fmt="png"):
    """Returns raw image bytes (png or jpg/jpeg) for the given figure."""
    return fig.to_image(format=fmt, scale=2)
