from flask import Flask, render_template, request, jsonify
import os
import json
import pandas as pd
from utils.tc_scraper import download_twitch_chat
from utils.llm_processor import generate_personas, summarize_comments, extract_useful_content

app = Flask(__name__)
app.config.from_pyfile('config.py')

# ========================
# TWITCH ROUTES
# ========================

@app.route('/')
def index():
    """Home page with options for Twitch chat analysis."""
    return render_template('index.html')

@app.route('/download_twitch', methods=['POST'])
def download_twitch():
    """Download chat from Twitch VODs."""
    twitch_urls = request.form.get('twitch_urls').split(',')
    results = {}
    
    for twitch_url in twitch_urls:
        twitch_url = twitch_url.strip()
        try:
            vod_id = twitch_url.split('/videos/')[-1].split('?')[0]
            success, message = download_twitch_chat(twitch_url, app.config['DATA_DIR'])
            results[vod_id] = {'success': success, 'message': message}
        except Exception as e:
            results[twitch_url] = {'success': False, 'message': str(e)}
    
    return jsonify(results)

@app.route('/generate_personas', methods=['POST'])
def create_personas():
    vod_id = request.form.get('vod_id')
    num_personas = int(request.form.get('num_personas', 3))
    
    comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
    if not os.path.exists(comments_path):
        return jsonify({'success': False, 'message': 'Chat data not found'})
    
    try:
        comments_df = pd.read_csv(comments_path)
        personas = generate_personas(comments_df, num_personas)
        summaries = summarize_comments(comments_df, personas)
        
        # Save personas
        persona_path = os.path.join(app.config['DATA_DIR'], 'personas', f'{vod_id}_personas.json')
        os.makedirs(os.path.dirname(persona_path), exist_ok=True)
        with open(persona_path, 'w') as f:
            json.dump(personas, f)
        
        # Save summaries
        summary_path = os.path.join(app.config['DATA_DIR'], 'summaries', f'{vod_id}_summaries.json')
        with open(summary_path, 'w') as f:
            json.dump(summaries, f)
        
        return jsonify({'success': True, 'personas': personas, 'summaries': summaries})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Processing failed: {e}'})

@app.route('/view_personas/<vod_id>')
def view_personas(vod_id):
    persona_path = os.path.join(app.config['DATA_DIR'], 'personas', f'{vod_id}_personas.json')
    summary_path = os.path.join(app.config['DATA_DIR'], 'summaries', f'{vod_id}_summaries.json')
    
    if not os.path.exists(persona_path) or not os.path.exists(summary_path):
        return render_template('error.html', message='Analysis data not found')
    
    with open(persona_path, 'r') as f:
        personas = json.load(f)
    with open(summary_path, 'r') as f:
        summaries = json.load(f)
    
    return render_template('personas.html', vod_id=vod_id, personas=personas, summaries=summaries)

@app.route('/negative_comments/<vod_id>')
def negative_comments(vod_id):
    comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
    
    if not os.path.exists(comments_path):
        return render_template('error.html', message='Chat data not found')
    
    comments_df = pd.read_csv(comments_path)
    sample_comments = comments_df.sample(min(10, len(comments_df)))
    
    processed_comments = [{
        'original': row['message'],
        'useful_content': extract_useful_content(row['message'])
    } for _, row in sample_comments.iterrows()]
    
    return render_template('negative_comments.html', 
                         vod_id=vod_id, 
                         comments=processed_comments)

@app.route('/list_vods')
def list_vods():
    chat_dir = os.path.join(app.config['DATA_DIR'], 'twitch_chat')
    if not os.path.exists(chat_dir):
        return jsonify([])
    
    vods = [f.replace('.csv', '') for f in os.listdir(chat_dir) if f.endswith('.csv')]
    return jsonify(vods)

# ========================
# INITIALIZATION
# ========================

if __name__ == '__main__':
    directories = ['twitch_chat', 'personas', 'summaries']
    for directory in directories:
        os.makedirs(os.path.join(app.config['DATA_DIR'], directory), exist_ok=True)
    app.run(debug=True)
