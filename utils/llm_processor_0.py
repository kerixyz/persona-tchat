import os
from openai import OpenAI
import pandas as pd
import json
import sys
from textblob import TextBlob
from flask import current_app
import re

from config import OPENAI_API_KEY, OPENAI_MODEL
api_key = OPENAI_API_KEY
client = OpenAI(api_key=api_key)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========================
# PERSONAS AND SUMMARIZATION
# ========================

def generate_personas(comments_df, num_personas=3):
    """Generate personas based on comments."""
    sample_size = min(100, len(comments_df))
    sample_comments = comments_df.sample(sample_size)['text'].tolist()
    
    # prompt = f"""
    # Based on the following {sample_size} Twitch comments, identify {num_personas} distinct personas or viewpoint clusters.
    # For each persona, provide:
    # 1. A short name/title
    # 2. A short 1 sentence description of their viewpoint 
    # 3. 3 direct quotes from the comments that exemplify their perspective.
    
    # Here are the comments:
    # {sample_comments}
    
    # Return the response as a JSON array of persona objects with 'name', 'description', and 'quotes' fields.
    # """
    prompt = f"""
    You are analyzing a sample of Twitch chat messages. Based on the following {sample_size} messages, identify {num_personas} distinct viewer personas, each representing a recognizable type of participant in Twitch chat.

    Each persona should reflect:
    - A distinct style of communication or behavioral pattern (e.g., hype spammer, advice-giver, donor, meme participant)
    - Their apparent motivations or goals in the chat (e.g., to support the streamer, to guide gameplay, to play with the community)
    - Common tone, frequency, or types of messages they post

    For each persona, provide:
    1. A short **name or title** (e.g., "The Strategist", "The Emote Engine")
    2. A **1–2 sentence description** of their behavior and communication style
    3. **Three representative quotes** from the provided messages that exemplify this persona

    Do not speculate beyond the content of the messages. Use only the sample below to infer patterns.

    Here are the Twitch chat messages:
    {sample_comments}

    Return your response as a **JSON array** of persona objects, each with the fields: `name`, `description`, and `quotes`.
    """


    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing social media comments and identifying distinct user personas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        try:
            personas = json.loads(content)
        except json.JSONDecodeError:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                personas_json = content[start_idx:end_idx]
                personas = json.loads(personas_json)
            else:
                # Default personas if parsing fails
                personas = []
                for i in range(num_personas):
                    personas.append({
                        "name": f"Persona {i+1}",
                        "description": f"Description for persona {i+1}"
                    })
        
        return personas
    
    except Exception as e:
        print(f"Error generating personas: {e}")
        # Handle other exceptions, e.g., by returning default personas
        personas = []
        for i in range(num_personas):
            personas.append({
                "name": f"Persona {i+1}",
                "description": f"Description for persona {i+1}",
                "quotes": ["quote 1", "quote 2", "quote 3"]
            })
        return personas

def summarize_comments(comments_df, personas): 
    summaries = {}
    
    sample_size = min(200, len(comments_df))
    sample_comments = comments_df.sample(sample_size)['text'].tolist()
    
    for persona in personas:
        prompt = f"""
        Summarize the following YouTube comments from the perspective of this persona:
        
        Persona: {persona['name']}
        Description: {persona['description']}
        Quotes: {', '.join(persona['quotes'])}
        
        Comments:
        {sample_comments}
        
        Provide a short summary (3-5 sentences) that captures what this persona would find most important or relevant in these comments.
        """
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in summarizing social media content from different perspectives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        summary = response.choices[0].message.content.strip()
        summaries[persona['name']] = summary
    
    return summaries

# ========================
# NEGATIVITY MANAGEMENT
# ========================

def analyze_negativity(comments_df, sample_size=100):
    col = 'message' if 'message' in comments_df.columns else 'text'
    comments = comments_df[col].dropna().tolist()

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    negative_comments = []

    for comment in comments:
        polarity = TextBlob(comment).sentiment.polarity
        if polarity < -0.1:
            sentiment_counts["Negative"] += 1
            negative_comments.append(comment)
        elif polarity > 0.1:
            sentiment_counts["Positive"] += 1
        else:
            sentiment_counts["Neutral"] += 1

    total = sum(sentiment_counts.values())
    sentiment_values = [
        round((sentiment_counts[label] / total) * 100, 1) if total > 0 else 0
        for label in sentiment_labels
    ]

    sample_comments = negative_comments[:sample_size]

    if not sample_comments:
        return {
            "examples": [],
            "sentiment_labels": sentiment_labels,
            "sentiment_values": sentiment_values
        }

    prompt = f"""
        You are a language expert analyzing potentially negative Twitch chat messages.

        Your task is to:
        - Rephrase harsh or unkind messages into constructive, neutral alternatives
        - Preserve the original meaning as much as possible
        - Leave already neutral or positive messages mostly unchanged

        Please return a JSON array where each item has:
        - `original`: the original comment
        - `neutralized`: the softened or more constructive version

        Here are the {sample_size} Twitch chat messages:
        {json.dumps(sample_comments, indent=2)}

        Respond with a JSON array only.
        """

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in transforming online chat messages into more constructive communication."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        try:
            examples = json.loads(content)
        except json.JSONDecodeError:
            start = content.find('[')
            end = content.rfind(']') + 1
            examples = json.loads(content[start:end]) if start >= 0 and end > start else []

        return {
            "examples": examples,
            "sentiment_labels": sentiment_labels,
            "sentiment_values": sentiment_values
        }

    except Exception as e:
        print(f"LLM negativity analysis failed: {e}")
        return {
            "examples": [],
            "sentiment_labels": sentiment_labels,
            "sentiment_values": sentiment_values
        }

# ========================
# COUNTERFACTUALS
# ========================
def generate_suggestions(comments_df, sample_size=100):
    from textblob import TextBlob

    col = 'message' if 'message' in comments_df.columns else 'text'
    comments = comments_df[col].dropna().tolist()

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    negative_comments = []

    for comment in comments:
        polarity = TextBlob(comment).sentiment.polarity
        if polarity < -0.1:
            sentiment_counts["Negative"] += 1
            negative_comments.append(comment)
        elif polarity > 0.1:
            sentiment_counts["Positive"] += 1
        else:
            sentiment_counts["Neutral"] += 1

    total = sum(sentiment_counts.values())
    sentiment_values = [
        round((sentiment_counts[label] / total) * 100, 1) if total > 0 else 0
        for label in sentiment_labels
    ]

    sample_comments = negative_comments[:sample_size]

    if not sample_comments:
        return {
            "examples": [],
            "sentiment_labels": sentiment_labels,
            "sentiment_values": sentiment_values
        }

    prompt = f"""
    You are reviewing negative Twitch chat messages and your task is to convert each into a **constructive suggestion** that could be genuinely helpful for the streamer.

    - Make the suggestion polite, practical, and game-related (if applicable).
    - Avoid sarcasm or vagueness. Focus on specific, actionable improvements.
    - Preserve the core intent (e.g., frustration about gameplay), but frame it as advice.

    Format: Return a JSON array with entries like:
    {{
      "original": "original message",
      "suggestion": "constructive counterfactual suggestion"
    }}

    Messages:
    {json.dumps(sample_comments, indent=2)}

    Respond with a JSON array only.
    """

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a Twitch coach transforming negative chat into helpful advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        try:
            examples = json.loads(content)
        except json.JSONDecodeError:
            start = content.find('[')
            end = content.rfind(']') + 1
            examples = json.loads(content[start:end]) if start >= 0 and end > start else []

        return {
            "examples": examples
        }

    except Exception as e:
        print(f"Error generating counterfactual suggestions: {e}")
        return {
            "examples": []
        }
