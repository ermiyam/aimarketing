from flask import Flask, request, jsonify, render_template
from advanced_ai_system import MarketingAI
import time

app = Flask(__name__)
marketing_ai = MarketingAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    data = request.get_json()
    query = data.get('query', '').lower()
    
    try:
        # Route to appropriate function based on query content
        if 'sentiment' in query or 'social media' in query:
            result = marketing_ai.analyze_social_media_sentiment(data)
        elif 'behavior' in query or 'customer' in query:
            result = marketing_ai.predict_customer_behavior(data)
        elif 'lead' in query or 'score' in query:
            result = marketing_ai.score_leads(data)
        elif 'campaign' in query or 'automate' in query:
            result = marketing_ai.automate_multichannel_campaign(data)
        elif 'brand' in query or 'monitor' in query:
            result = marketing_ai.monitor_brand(data)
        elif 'feedback' in query:
            result = marketing_ai.analyze_customer_feedback(data)
        else:
            # Default to general analysis
            result = {
                'message': 'Could not determine specific analysis type. Please be more specific.',
                'query': query
            }
        
        # Add metrics for dashboard
        result.update({
            'sentiment': result.get('sentiment', 0.0),
            'engagement': result.get('engagement', 0.0),
            'leadScore': result.get('leadScore', 0.0)
        })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'sentiment': 0.0,
            'engagement': 0.0,
            'leadScore': 0.0
        }), 500

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    result = marketing_ai.get_system_status()
    return jsonify(result)

@app.route('/api/visualize-metrics', methods=['GET'])
def visualize_metrics():
    result = marketing_ai.visualize_metrics()
    return jsonify(result)

if __name__ == '__main__':
    print("Starting Unified Marketing AI Server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 