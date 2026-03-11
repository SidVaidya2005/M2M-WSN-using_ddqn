import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from train_final_ddqn import train_final_ddqn

app = Flask(__name__)

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

@app.route('/run_training', methods=['POST'])
def run_training():
    data = request.json
    try:
        episodes = int(data.get('episodes', 100))
        nodes = int(data.get('nodes', 550))
        seed = int(data.get('seed', 42))
        lr = float(data.get('lr', 1e-4))
        gamma = float(data.get('gamma', 0.99))
        batch_size = int(data.get('batch_size', 64))
        death_threshold = float(data.get('death_threshold', 0.3))
        
        # Run training (This will take some time)
        agent, results = train_final_ddqn(
            episodes=episodes,
            seed=seed,
            N=nodes,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            death_threshold=death_threshold
        )
        
        # Always bust cache by adding a query param timestamp in frontend
        return jsonify({
            'status': 'success',
            'message': 'Training completed successfully.',
            'image_url': '/results/final_ddqn_training.png',
            'gif_url': '/results/final_ddqn_best_episode.gif',
            'results': results
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
