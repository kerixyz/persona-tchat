from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import pandas as pd
from utils.tc_scraper import download_twitch_chat
import nltk
from utils.llm_processor import process_comments

nltk.download('wordnet')

app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route('/')
def index():
    """Home page with options for Twitch chat analysis."""
    return render_template('index.html')


@app.route('/results/<vod_id>')
def show_results(vod_id):
    # Load all analysis data
    analysis_data = {
        'personas': load_json(f'personas/{vod_id}_personas.json'),
        'summaries': load_json(f'summaries/{vod_id}_summaries.json'),
        'negativity': load_json(f'negativity/{vod_id}_negativity.json'),
        'suggestions': load_json(f'suggestions/{vod_id}_suggestions.json')
    }
    return render_template('results.html', vod_id=vod_id, **analysis_data)

def load_json(relative_path):
    path = os.path.join(app.config['DATA_DIR'], relative_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@app.route('/download_twitch', methods=['POST'])
def download_twitch():
    try:
        twitch_url = request.form.get('twitch_urls').strip()
        vod_id = twitch_url.split('/videos/')[-1].split('?')[0]
        
        success, message = download_twitch_chat(twitch_url, app.config['DATA_DIR'])

        if not success:
            return jsonify({'success': False, 'message': message}) if request.headers.get('X-Requested-With') == 'XMLHttpRequest' else render_template('error.html', message=message)

        comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
        comments_df = pd.read_csv(comments_path)

        result = process_comments(comments_df)

        # Extract components
        personas = result.get("personas", [])
        summaries = {
            "overall_summary": " ".join(result.get("summaries", {}).values()),
            "total_messages": len(comments_df),
            "unique_users": comments_df['user_id'].nunique() if 'user_id' in comments_df.columns else "N/A",
            "avg_sentiment": "N/A"
        }
        negativity = {
            "examples": result.get("negativity", []),
            "sentiment_labels": ["Negative", "Neutral", "Positive"],
            "sentiment_values": [0, 0, 0]
        }
        suggestions = [
            {"original": c["original"], "suggestion": c["suggestion"]}
            for c in result.get("negativity", []) if c.get("suggestion")
        ]

        save_analysis(vod_id, {
            'personas': [
                {
                    "name": p["name"],
                    "description": p["description"],
                    "feedback": result.get("summaries", {}).get(p["name"], "").split(". ")[:3]
                } for p in personas
            ],
            'summaries': summaries,
            'negativity': negativity,
            'suggestions': suggestions
        })

        return jsonify({'success': True, 'redirect_url': url_for('show_results', vod_id=vod_id)}) if request.headers.get('X-Requested-With') == 'XMLHttpRequest' else redirect(url_for('show_results', vod_id=vod_id))

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}) if request.headers.get('X-Requested-With') == 'XMLHttpRequest' else render_template('error.html', message=str(e))
    
def save_analysis(vod_id, data):
    for key in ['personas', 'summaries', 'negativity', 'suggestions']:
        path = os.path.join(app.config['DATA_DIR'], key, f'{vod_id}_{key}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data[key], f)

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    directories = ['twitch_chat', 'personas', 'summaries']
    for directory in directories:
        os.makedirs(os.path.join(app.config['DATA_DIR'], directory), exist_ok=True)
    app.run(debug=True)
