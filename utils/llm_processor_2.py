import os
import json
import pandas as pd
from textblob import TextBlob
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

# ========= LLM Client =========
client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm(messages, model=OPENAI_MODEL, temperature=0.7, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""


# ========= Persona Generator =========
def generate_personas(comments_df, num_personas=3, sample_size=100, model=OPENAI_MODEL):
    sample_comments = comments_df.sample(min(sample_size, len(comments_df)))['text'].tolist()

    prompt = f"""
    Analyze these {len(sample_comments)} Twitch chat messages. Identify {num_personas} distinct personas:

    For each persona:
    1. A name (e.g., \"The Hype Engine\")
    2. A 1-2 sentence description
    3. 3 direct quotes that illustrate the persona

    Comments:
    {sample_comments}

    Return a JSON array with keys: `name`, `description`, and `quotes`.
    """

    messages = [
        {"role": "system", "content": "You are skilled at extracting behavioral archetypes from user-generated text."},
        {"role": "user", "content": prompt}
    ]
    response = call_llm(messages, model=model, max_tokens=1500)

    try:
        start, end = response.find('['), response.rfind(']') + 1
        return json.loads(response[start:end])
    except:
        return [{"name": f"Persona {i+1}", "description": "Unknown", "quotes": []} for i in range(num_personas)]


# ========= Comment Summarizer =========
def summarize_comments(comments_df, personas, sample_size=200, model=OPENAI_MODEL):
    sample_comments = comments_df.sample(min(sample_size, len(comments_df)))['text'].tolist()
    summaries = {}

    for persona in personas:
        prompt = f"""
        Summarize these Twitch chat comments from the point of view of:

        Persona: {persona['name']}
        Description: {persona['description']}
        Example quotes: {', '.join(persona['quotes'])}

        Comments:
        {sample_comments}

        Summary (3-5 sentences):
        """
        messages = [{"role": "user", "content": prompt}]
        summary = call_llm(messages, model=model)
        summaries[persona['name']] = summary

    return summaries


# ========= Negativity Analyzer =========
NEUTRALIZE_MAP = {
    "hate": "don't prefer",
    "awful": "not ideal",
    "terrible": "could use improvement",
    "worst": "less than ideal",
    "stupid": "unwise",
    "horrible": "not great",
    "disgusting": "unpleasant",
    "useless": "not very helpful",
    "annoying": "a bit bothersome",
    "boring": "not very interesting",
    "dumb": "unwise",
    "sucks": "could be better",
    "fail": "didn't go as planned",
    "can't stand": "don't enjoy",
    "problem": "issue",
    "bad": "not ideal",
    "lame": "not very exciting"
}

def neutralize_manual(text):
    for word, repl in NEUTRALIZE_MAP.items():
        text = text.replace(word, repl)
    return text

def neutralize_llm(text, model=OPENAI_MODEL):
    prompt = f"""Transform the following message into a neutral, constructive version:

    \"{text}\"
    """
    messages = [
        {"role": "system", "content": "You're a helpful assistant skilled at rephrasing negative comments into constructive feedback."},
        {"role": "user", "content": prompt}
    ]
    return call_llm(messages, model=model, temperature=0.5, max_tokens=150)

def neutralize_text(text, method='llm', model=OPENAI_MODEL):
    if method == 'llm':
        return neutralize_llm(text, model=model)
    return neutralize_manual(text)

def analyze_negativity(comments_df, method='llm', model=OPENAI_MODEL):
    results = {"examples": [], "Negative": 0, "Neutral": 0, "Positive": 0}

    if 'text' not in comments_df.columns:
        print("Error: 'message' column not found in DataFrame")
        return results

    for msg in comments_df['text'].dropna():
        polarity = TextBlob(str(msg)).sentiment.polarity
        label = "Negative" if polarity < -0.1 else "Positive" if polarity > 0.1 else "Neutral"
        results[label] += 1

        if label == "Negative":
            revised = neutralize_text(msg, method=method, model=model)
            if revised != msg:
                results["examples"].append({"original": msg, "neutralized": revised, "polarity": round(polarity, 2)})

    return results


# ========= Counterfactual Suggester =========
def extract_useful_feedback(comment, model=OPENAI_MODEL):
    prompt = f"""
    Extract useful feedback from this negative Twitch comment. Ignore unhelpful language.

    Comment: \"{comment}\"

    Useful content:
    """
    return call_llm([{"role": "user", "content": prompt}], model=model, temperature=0.3)

def suggest_counterfactual(useful_content, model=OPENAI_MODEL):
    prompt = f"""
    Based on this constructive feedback, suggest one actionable counterfactual (what could have been done instead):

    \"{useful_content}\"

    Suggestion:
    """
    return call_llm([{"role": "user", "content": prompt}], model=model, temperature=0.4)

def generate_suggestions(negative_examples, model=OPENAI_MODEL):
    suggestions = []
    for ex in negative_examples:
        useful = extract_useful_feedback(ex["original"], model=model)
        suggestion = suggest_counterfactual(useful, model=model)
        suggestions.append({"original": ex["original"], "useful": useful, "suggestion": suggestion})
    return suggestions