import os
from openai import OpenAI
import pandas as pd
import json
import sys
from textblob import TextBlob
from flask import current_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL
api_key = OPENAI_API_KEY
client = OpenAI(api_key=api_key)

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
        
        # Attempt to parse JSON directly if possible
        try:
            personas = json.loads(content)
        except json.JSONDecodeError:
            # If not directly parseable, try to extract JSON from the content
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

def neutralize_text(text, method='manual'):
    """Unified neutralization entry point"""
    if method == 'llm':
        return _neutralize_with_llm(text)
    return _neutralize_manual(text)

def _neutralize_manual(text):
    """Original replacement-based approach"""
    for neg, neu in NEUTRALIZE_MAP.items():
        text = text.replace(neg, neu)
    return text

def _neutralize_with_llm(text):
    """LLM-powered psychological reframing"""
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "system",
                "content": """Transform negative messages using evidence-based strategies: Neutral term substitution

                Maintain original intent while improving constructive value."""
            }, {
                "role": "user", 
                "content": text
            }],
            temperature=0.7,
            max_tokens=150,
            request_timeout=15  # Prevent hanging
        )
       
        return response.choices[0].message['content'].strip()
    except Exception as e:
        current_app.logger.error(f"LLM Neutralization failed: {e}")
        return _neutralize_manual(text)  # Fallback to manual
    
def analyze_negativity(comments_df, method=None):
    method = method or current_app.config['NEUTRALIZE_METHOD']
    
    analysis = {
        "method_used": method,
        "examples": [],
        "sentiment_labels": ["Negative", "Neutral", "Positive"],
        "sentiment_values": [0, 0, 0]
    }

    if 'message' not in comments_df.columns:
        return analysis

    sentiments = []
    for _, row in comments_df.iterrows():
        msg = str(row['message'])
        blob = TextBlob(msg)
        polarity = blob.sentiment.polarity
        
        if polarity < -0.1:  # Adjusted threshold
            neutralized = neutralize_text(msg, method)
            
            if neutralized != msg:  # Only track meaningful changes
                analysis["examples"].append({
                    "original": msg,
                    "neutralized": neutralized,
                    "polarity": round(polarity, 2)
                })
            
            sentiment = "Negative"
        elif polarity > 0.1:
            sentiment = "Positive"
        else:
            sentiment = "Neutral"
            
        sentiments.append(sentiment)

    # Calculate percentages
    counts = pd.Series(sentiments).value_counts(normalize=True) * 100
    for i, label in enumerate(analysis["sentiment_labels"]):
        analysis["sentiment_values"][i] = round(counts.get(label, 0), 1)

    # Sort by most negative first
    analysis["examples"] = sorted(
        [ex for ex in analysis["examples"] if ex["neutralized"] != ex["original"]],
        key=lambda x: x["polarity"]
    )[:5]

    return analysis


# ========================
# COUNTERFACTUALS
# ========================

def generate_suggestions(comments_df):
    """Temporary placeholder for suggestions"""
    return [
        {
            "title": "Engage with Frequent Chatters",
            "description": "Consider highlighting regular participants in your next stream",
            "category": "Community Building"
        }
    ]


def extract_useful_content(negative_comment):
    
    prompt = f"""
    The following is a negative comment from Twitch Chat. Extract any useful feedback or constructive criticism from it,
    ignoring toxic or unhelpful parts. If there is no useful content, indicate that.
    
    Comment: "{negative_comment}"
    
    Useful content:
    """
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in extracting constructive feedback from negative comments."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    useful_content = response.choices[0].message.content.strip()
    return useful_content
