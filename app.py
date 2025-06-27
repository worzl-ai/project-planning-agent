"""
Project Planning Agent - Foundation Agent
"""
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'agent': 'project-planning-agent',
        'status': 'operational',
        'version': '1.0.0',
        'capabilities': [
            'project_planning',
            'roadmap_creation',
            'task_breakdown',
            'brd_foundation'
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'agent': 'project-planning-agent',
        'version': '1.0.0',
        'architecture': 'modern',
        'features': {
            'oauth': False,  # Will be configured
            'mcp': True,
            'cards': True,
            'a2a': True,
            'brd_integration': True
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
