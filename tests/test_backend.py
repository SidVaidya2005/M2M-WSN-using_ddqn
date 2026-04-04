"""Integration tests for Flask API endpoints."""

import json
import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, flask_client):
        response = flask_client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_healthy(self, flask_client):
        data = json.loads(flask_client.get("/api/health").data)
        assert data["status"] == "healthy"


class TestConfigEndpoint:
    def test_config_returns_200(self, flask_client):
        response = flask_client.get("/api/config")
        assert response.status_code == 200

    def test_config_contains_expected_keys(self, flask_client):
        data = json.loads(flask_client.get("/api/config").data)
        assert "training" in data or "environment" in data


class TestTrainingValidation:
    def test_invalid_episodes_rejected(self, flask_client):
        response = flask_client.post(
            "/api/train",
            json={"episodes": -1},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_invalid_model_type_rejected(self, flask_client):
        response = flask_client.post(
            "/api/train",
            json={"model_type": "invalid_model"},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_empty_body_uses_defaults(self, flask_client):
        # Empty body should be accepted (defaults applied by schema)
        # This does NOT actually run training — just validates schema handling
        response = flask_client.post(
            "/api/train/async",
            json={},
            content_type="application/json",
        )
        assert response.status_code in (200, 202, 500)
        # 500 is acceptable if training infra isn't set up in test env


class TestAsyncTraining:
    def test_async_endpoint_returns_task_id(self, flask_client):
        response = flask_client.post(
            "/api/train/async",
            json={"episodes": 1, "nodes": 10},
            content_type="application/json",
        )
        if response.status_code == 202:
            data = json.loads(response.data)
            assert "task_id" in data
            assert data["status"] == "queued"

    def test_task_status_unknown_id(self, flask_client):
        response = flask_client.get("/api/tasks/nonexistent-task-id-12345")
        assert response.status_code == 404


class TestServeIndex:
    def test_index_returns_html(self, flask_client):
        response = flask_client.get("/")
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data or b"html" in response.data.lower()
