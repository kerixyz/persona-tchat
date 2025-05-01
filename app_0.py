from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import pandas as pd
from utils.tc_scraper import download_twitch_chat
from utils.llm_processor import generate_personas, summarize_comments, analyze_negativity, generate_suggestions

import nltk
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
    """Handle both AJAX and form submissions"""
    try:
        twitch_url = request.form.get('twitch_urls').strip()
        vod_id = twitch_url.split('/videos/')[-1].split('?')[0]
        
        success, message = download_twitch_chat(twitch_url, app.config['DATA_DIR'])

        if not success:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': message})
            else:
                return render_template('error.html', message=message)
            
        comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
        comments_df = pd.read_csv(comments_path)

        personas = generate_personas(comments_df, 3)
        summaries = summarize_comments(comments_df, personas)
        negativity = analyze_negativity(comments_df, sample_size=100)   
        suggestions = generate_suggestions(comments_df) 

        save_analysis(vod_id, {
            'personas': personas,
            'summaries': summaries,
            'negativity': negativity,
            'suggestions': suggestions
        })

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'redirect_url': url_for('show_results', vod_id=vod_id)})
        else:
            return redirect(url_for('show_results', vod_id=vod_id))
        
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': str(e)})
        else:
            return render_template('error.html', message=str(e))

def save_analysis(vod_id, data):
    for key in ['personas', 'summaries', 'negativity', 'suggestions']:
        path = os.path.join(app.config['DATA_DIR'], key, f'{vod_id}_{key}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data[key], f)


# @app.route('/generate_personas', methods=['POST'])
# def create_personas():
#     vod_id = request.form.get('vod_id')
#     num_personas = int(request.form.get('num_personas', 3))
    
#     comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
#     if not os.path.exists(comments_path):
#         return jsonify({'success': False, 'message': 'Chat data not found'})
    
#     try:
#         comments_df = pd.read_csv(comments_path)
#         personas = generate_personas(comments_df, num_personas)
#         summaries = summarize_comments(comments_df, personas)
        
#         persona_path = os.path.join(app.config['DATA_DIR'], 'personas', f'{vod_id}_personas.json')
#         os.makedirs(os.path.dirname(persona_path), exist_ok=True)
#         with open(persona_path, 'w') as f:
#             json.dump(personas, f)
        
#         summary_path = os.path.join(app.config['DATA_DIR'], 'summaries', f'{vod_id}_summaries.json')
#         with open(summary_path, 'w') as f:
#             json.dump(summaries, f)
        
#         return jsonify({'success': True, 'personas': personas, 'summaries': summaries})
#     except Exception as e:
#         return jsonify({'success': False, 'message': f'Processing failed: {e}'})

# @app.route('/view_personas/<vod_id>')
# def view_personas(vod_id):
#     persona_path = os.path.join(app.config['DATA_DIR'], 'personas', f'{vod_id}_personas.json')
#     summary_path = os.path.join(app.config['DATA_DIR'], 'summaries', f'{vod_id}_summaries.json')
    
#     if not os.path.exists(persona_path) or not os.path.exists(summary_path):
#         return render_template('error.html', message='Analysis data not found')
    
#     with open(persona_path, 'r') as f:
#         personas = json.load(f)
#     with open(summary_path, 'r') as f:
#         summaries = json.load(f)
    
#     return render_template('personas.html', vod_id=vod_id, personas=personas, summaries=summaries)

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
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    directories = ['twitch_chat', 'personas', 'summaries']
    for directory in directories:
        os.makedirs(os.path.join(app.config['DATA_DIR'], directory), exist_ok=True)
    app.run(debug=True)
