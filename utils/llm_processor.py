import json
from openai import OpenAI
from textblob import TextBlob
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def process_comments(comments_df, num_personas=3, sample_size=200):
    """Single-call processor for personas, summaries, and negativity handling."""

    col = 'message' if 'message' in comments_df.columns else 'text'
    comments = comments_df[col].dropna().tolist()

    sample_comments = comments[:sample_size]
    negative_comments = [c for c in sample_comments if TextBlob(c).sentiment.polarity < -0.1]

    # Basic sentiment analysis
    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for comment in sample_comments:
        polarity = TextBlob(comment).sentiment.polarity
        if polarity < -0.1:
            sentiment_counts["Negative"] += 1
        elif polarity > 0.1:
            sentiment_counts["Positive"] += 1
        else:
            sentiment_counts["Neutral"] += 1

    total = sum(sentiment_counts.values())
    sentiment_values = [
        round((sentiment_counts["Negative"] / total) * 100, 1) if total > 0 else 0,
        round((sentiment_counts["Neutral"] / total) * 100, 1) if total > 0 else 0,
        round((sentiment_counts["Positive"] / total) * 100, 1) if total > 0 else 0,
    ]


    prompt = f"""
            You are an expert in analyzing Twitch chat messages.

            Given the following sample of chat messages, perform the following:

            1. Identify {num_personas} distinct viewer personas:
            - For each, return:
                - `name`: short label
                - `description`: 1–2 sentence summary of their behavior
                - `quotes`: 3 example messages

            2. For each persona, summarize their perspective on the chat (3–5 sentences).

            3. Rephrase negative messages:
            - For each, return:
                - `original`
                - `neutralized`: softened version
                - `suggestion`: a polite, constructive improvement if possible

            Sample Messages:
            {json.dumps(sample_comments, indent=2)}

            Negative Messages:
            {json.dumps(negative_comments, indent=2)}

            Respond with a JSON object:
            {{
            "personas": [
                {{ "name": "...", "description": "...", "quotes": ["..."] }}
            ],
            "summaries": {{
                "Persona Name": "Summary..."
            }},
            "negativity": [
                {{
                "original": "...",
                "neutralized": "...",
                "suggestion": "..."
                }}
            ]
            }}
            """

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You analyze Twitch chats and extract structured insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=3000
        )
        content = response.choices[0].message.content

        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            return json.loads(content[start:end])
        except Exception as e:
            print("Error parsing JSON from LLM:", e)
            return {}
    except Exception as e:
        print("LLM call failed:", e)
        return {}
    
    
