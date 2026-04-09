# API Reference

## REST API

All endpoints are at `/api/` base path.

### Authentication

Currently no authentication required. For production, add:

```python
# backend/app.py
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

---

## Endpoints

### GET /api/health

Health check endpoint to verify server is running.

**Response:**

```json
{
  "status": "healthy"
}
```

**Status Code:** 200

**Example:**

```bash
curl http://localhost:5001/api/health
```

---

### GET /api/config

Retrieve current configuration.

**Response:**

```json
{
  "training": {
    "episodes": 100,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "gamma": 0.99
  },
  "environment": {
    "num_nodes": 50,
    "arena_size": [500, 500],
    "max_steps": 1000
  },
  "paths": {
    "models": "results/models",
    "metrics": "results/metrics"
  }
}
```

**Status Code:** 200

**Example:**

```bash
curl http://localhost:5001/api/config
```

---

### POST /api/train

Start training a new model.

**Request Body:**

```json
{
  "episodes": 100,
  "nodes": 50,
  "model_type": "ddqn",
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "batch_size": 64,
  "seed": 42
}
```

| Field         | Type  | Default | Range        |
| ------------- | ----- | ------- | ------------ |
| episodes      | int   | 100     | 1-10000      |
| nodes         | int   | 50      | 10-10000     |
| model_type    | str   | ddqn    | dqn or ddqn  |
| learning_rate | float | 1e-4    | 1e-6 to 1e-1 |
| gamma         | float | 0.99    | 0.0-1.0      |
| batch_size    | int   | 64      | 8-512        |
| seed          | int   | 42      | any int      |

**Response (Success):**

```json
{
  "status": "success",
  "message": "Training completed successfully with DDQN.",
  "episodes": 100,
  "nodes": 50,
  "model_type": "ddqn",
  "mean_reward": 145.32,
  "max_reward": 180.5,
  "results": {
    "best_lifetime": 180.5,
    "best_episode": 73,
    "avg_lifetime_final_10": 172.4
  },
  "model_path": "results/models/run_20260406_080528_model.pth"
}
```

`results.best_episode` is the 1-based episode number where the maximum reward occurred.
`results.avg_lifetime_final_10` is the average reward over the last up to 10 episodes.

**Status Code:** 200

**Response (Error):**

```json
{
  "status": "error",
  "message": "Invalid batch_size: must be >= 8"
}
```

**Status Code:** 400 (bad request) or 500 (server error)

**Note:** This endpoint blocks until training completes. For async training, use background jobs (see async section).

**Example:**

```bash
curl -X POST http://localhost:5001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": 50,
    "nodes": 100,
    "model_type": "dqn",
    "learning_rate": 0.0001
  }'
```

---

### POST /api/train/async

Start training without blocking. Returns a `task_id` immediately; poll for status.

**Request Body:** Same fields as `POST /api/train`.

**Response:**

```json
{ "task_id": "550e8400-e29b-41d4-a716-446655440000" }
```

**Status Code:** 202

---

### GET /api/tasks/\<task_id\>

Poll the status of an async training or benchmark job.

**Response:**

```json
{ "status": "queued" }
{ "status": "running" }
{ "status": "completed", "result": { ... } }
{ "status": "failed",    "error": "..." }
{ "status": "not_found" }
```

`"not_found"` is returned (not a 404) if the task_id is unknown or the server was restarted (task registry is in-memory only).

**Status Code:** 200

---

### GET /api/history

Return all training run metadata, newest first. Scans `results/metrics/` for `*_metadata.json` files. If a matching `*_evaluation.json` exists for a run, it is inlined as `run["evaluation"]`.

**Response:**

```json
[
  {
    "run_id": "run_20260406_080528",
    "timestamp": "2026-04-06T08:06:10.290101",
    "config": { "model_type": "ddqn", "episodes": 100, "nodes": 550, ... },
    "metrics": { "mean_reward": 145.32, "max_reward": 180.5, ... },
    "image_url": "/api/visualizations/run_20260406_080528_plot.png",
    "evaluation": { ... }
  }
]
```

**Status Code:** 200

---

### POST /api/evaluate

Submit an async baseline benchmark job for a completed training run.

**Request Body:**

```json
{ "run_id": "run_20260406_080528", "episodes": 10 }
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| run_id | str | required | From `GET /api/history` |
| episodes | int | 10 | 1–100 |

**Response:**

```json
{ "task_id": "550e8400-e29b-41d4-a716-446655440000" }
```

Poll `GET /api/tasks/<task_id>` for results. On completion, writes `results/metrics/{run_id}_evaluation.json`.

**Status Code:** 202

---

### GET /api/results/<filename>

Retrieve output files (models, metrics, visualizations).

**Parameters:**

- `filename`: Name of file in results/ directory

**Response:** File content (binary for .pth, JSON for .json, PNG for .png)

**Status Code:**

- 200: File found
- 404: File not found

**Example:**

```bash
# Download JSON metrics
curl http://localhost:5001/api/results/training_metrics.json > metrics.json

# Download trained model
curl http://localhost:5001/api/results/trained_model.pth > model.pth
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

### Common Status Codes

| Code | Meaning      | Example               |
| ---- | ------------ | --------------------- |
| 200  | Success      | Training completed    |
| 400  | Bad request  | Invalid parameter     |
| 404  | Not found    | Results file missing  |
| 500  | Server error | Crash during training |

---

## Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:5001/api"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())  # {"status": "healthy"}

# Get config
response = requests.get(f"{BASE_URL}/config")
config = response.json()
print(f"Current episodes: {config['training']['episodes']}")

# Start training
training_config = {
    "episodes": 100,
    "nodes": 50,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 64,
}

response = requests.post(
    f"{BASE_URL}/train",
    json=training_config,
)

if response.status_code == 200:
    result = response.json()
    print(f"Training complete!")
    print(f"Mean reward: {result['mean_reward']:.2f}")
else:
    print(f"Error: {response.json()}")

# Download results
response = requests.get(f"{BASE_URL}/results/training_metrics.json")
metrics = response.json()
print(json.dumps(metrics, indent=2))
```

---

## JavaScript Client Example

```javascript
const BASE_URL = "http://localhost:5001/api";

// Health check
async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  const data = await response.json();
  console.log(data); // {status: "healthy"}
}

// Start training
async function startTraining(trainingConfig) {
  const response = await fetch(`${BASE_URL}/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(trainingConfig),
  });

  if (response.ok) {
    const result = await response.json();
    console.log(`Training complete!`);
    console.log(`Mean reward: ${result.mean_reward.toFixed(2)}`);
  } else {
    console.error(`Error: ${await response.text()}`);
  }
}

// Download metrics
async function downloadResults(filename) {
  const response = await fetch(`${BASE_URL}/results/${filename}`);
  const blob = await response.blob();

  // Create download link
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
}

// Usage
const config = {
  episodes: 100,
  nodes: 50,
  learning_rate: 0.0001,
  gamma: 0.99,
};
await startTraining(config);
```

---

## Input Validation (Schemas)

The API validates all inputs using Marshmallow schemas defined in `backend/schemas.py`.

```python
# backend/schemas.py
from marshmallow import Schema, fields, validate

class TrainingRequestSchema(Schema):
    episodes = fields.Int(
        required=False,
        validate=validate.Range(min=1, max=10000),
        missing=100,
    )
    nodes = fields.Int(
        required=False,
        validate=validate.Range(min=10, max=10000),
        missing=50,
    )
    learning_rate = fields.Float(
        required=False,
        validate=validate.Range(min=1e-6, max=1e-1),
        missing=1e-4,
    )
```

Invalid requests return:

```json
{
  "error": "Validation error",
  "details": {
    "learning_rate": ["Must be between 1e-6 and 1e-1"]
  }
}
```

---

## Deployment

### Running with Gunicorn (Production)

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### Running with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w 4", "-b 0.0.0.0:5000", "backend.app:app"]
```

Build and run:

```bash
docker build -t wsn-ddqn .
docker run -p 5000:5000 wsn-ddqn
```

---

## Rate Limiting (Optional Enhancement)

Add rate limiting to prevent abuse:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app, key_func=get_remote_address)

@api_bp.route("/train", methods=["POST"])
@limiter.limit("1 per hour")
def train_model():
    ...
```

---

## Monitoring & Logging

All API calls are logged. Configure in `config/logging_config.yaml`:

```yaml
loggers:
  backend:
    level: INFO
    handlers: [console, file]
```

View logs:

```bash
tail -f logs/app.log
```
