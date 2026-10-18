import os
import json
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

DATA_FILE = 'data/eval_results/evaluation_data.json'
IMAGES_DIR = 'data/processed'

# Ensure directories exist
os.makedirs('data/eval_results', exist_ok=True)

# Generate mock data if not exists
if not os.path.exists(DATA_FILE):
    mock_data = []
    # Try to find some images to use
    if os.path.exists(IMAGES_DIR):
        images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))][:50]
        for i, img in enumerate(images):
            mock_data.append({
                "id": i + 1,
                "image_filename": img,
                "baseline_caption": "The shelf is mostly empty. There are some products.",
                "mop_caption": f"Kanops Taxonomy: Unknown Retailer. Products detected: Various Grocery items.",
                "annotations": None
            })
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=4)

def load_data():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/task', methods=['GET'])
def get_task():
    user = request.args.get('user')
    if not user:
        return jsonify({"error": "User not specified"}), 400
        
    data = load_data()
    
    # Load translations if available
    translations = {}
    trans_file = 'data/eval_results/translations_kr.json'
    if os.path.exists(trans_file):
        with open(trans_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
            
    # Find next unannotated for this user
    for item in data:
        # Initialize if missing or None
        if not isinstance(item.get('annotations'), dict):
            item['annotations'] = {}
            
        if user not in item['annotations']:
            # Inject Korean translations
            img_id = str(item['id'])
            if img_id in translations:
                item['baseline_caption_kr'] = translations[img_id].get('baseline_caption')
                item['mop_caption_kr'] = translations[img_id].get('mop_caption')
            return jsonify(item)
    return jsonify({"status": "complete", "message": "All annotations completed!"})

@app.route('/api/annotate', methods=['POST'])
def annotate():
    annotation = request.json
    item_id = annotation.get('id')
    user = annotation.get('user')
    
    if not user:
        return jsonify({"error": "User not specified"}), 400
        
    data = load_data()
    for item in data:
        if item['id'] == item_id:
            if not isinstance(item.get('annotations'), dict):
                item['annotations'] = {}
                
            item['annotations'][user] = {
                "winner": annotation.get('winner'),
                "notes": annotation.get('notes', '')
            }
            break
            
    save_data(data)
    return jsonify({"success": True})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    user = request.args.get('user')
    data = load_data()
    total = len(data)
    
    if user:
        completed = sum(1 for item in data if isinstance(item.get('annotations'), dict) and user in item['annotations'])
    else:
        completed = 0
        
    return jsonify({"total": total, "completed": completed})

@app.route('/images/<filename>')
def serve_image(filename):
    path = os.path.join(os.getcwd(), IMAGES_DIR, filename)
    if os.path.exists(path):
        return send_file(path)
    return "Image not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
