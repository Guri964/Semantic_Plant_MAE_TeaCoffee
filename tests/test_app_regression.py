"""Application regression tests (preserve existing functionality).

Runs against mock models with the real Flask routes, DB, templates, color
pipeline and KB. Real-weight behavior is covered by test_model_integration.py.
"""

import io
import os
import sqlite3

import pytest
import torch

from conftest import (
    make_healthy_leaf, make_yellow_patch_leaf, make_non_leaf, save_img,
    MockCropModelWithFeatures, MockHealthModel,
)


def _upload(client, path, field='image'):
    with open(path, 'rb') as f:
        data = {field: (io.BytesIO(f.read()), os.path.basename(path))}
        return client.post('/predict', data=data, content_type='multipart/form-data')


class TestAuth:
    def test_signup_login_logout(self, client):
        r = client.post('/signup', json={
            'username': 'alice', 'password': 'secret123', 'confirm_password': 'secret123'})
        assert r.status_code == 200 and r.get_json()['success']
        r = client.post('/login', json={'username': 'alice', 'password': 'secret123'})
        assert r.status_code == 200 and r.get_json()['success']
        r = client.get('/logout', follow_redirects=False)
        assert r.status_code in (302, 303)

    def test_login_wrong_password(self, client):
        client.post('/signup', json={
            'username': 'bob', 'password': 'secret123', 'confirm_password': 'secret123'})
        r = client.post('/login', json={'username': 'bob', 'password': 'wrong'})
        assert r.status_code == 401

    def test_signup_password_mismatch(self, client):
        r = client.post('/signup', json={
            'username': 'c', 'password': 'secret123', 'confirm_password': 'other'})
        assert r.status_code == 400

    def test_duplicate_signup(self, client):
        payload = {'username': 'dup', 'password': 'secret123', 'confirm_password': 'secret123'}
        client.post('/signup', json=payload)
        r = client.post('/signup', json=payload)
        assert r.status_code == 409


class TestDashboard:
    def test_requires_login(self, client):
        r = client.get('/dashboard', follow_redirects=False)
        assert r.status_code in (302, 303)
        assert '/login' in r.headers['Location']

    def test_renders_when_logged_in(self, logged_in_client):
        r = logged_in_client.get('/dashboard')
        assert r.status_code == 200
        assert b'Diagnose a leaf' in r.data or b'dashboard' in r.data.lower()

    def test_index_renders(self, client):
        assert client.get('/').status_code == 200


class TestUploadValidation:
    def test_no_file(self, logged_in_client):
        r = logged_in_client.post('/predict', data={})
        assert r.status_code == 400

    def test_bad_extension(self, logged_in_client, tmp_path):
        p = os.path.join(str(tmp_path), 'file.txt')
        open(p, 'w').write('hello')
        r = _upload(logged_in_client, p)
        assert r.status_code == 400

    def test_corrupt_image(self, logged_in_client, tmp_path):
        p = os.path.join(str(tmp_path), 'corrupt.jpg')
        open(p, 'wb').write(b'not an image at all')
        r = _upload(logged_in_client, p)
        assert r.status_code == 400
        assert r.get_json()['reason'] == 'invalid_image'

    def test_non_leaf_rejected(self, logged_in_client, tmp_path):
        p = save_img(tmp_path, make_non_leaf(), 'background.png')
        r = _upload(logged_in_client, p)
        assert r.status_code == 200
        body = r.get_json()
        assert body['success'] is False
        assert body['reason'] == 'not_a_leaf'

    def test_predict_requires_login(self, client, healthy_image_path):
        r = _upload(client, healthy_image_path)
        assert r.status_code in (302, 303)


class TestPredictionFlow:
    def test_healthy_leaf_consistent_output(self, logged_in_client, healthy_image_path):
        r = _upload(logged_in_client, healthy_image_path)
        assert r.status_code == 200
        body = r.get_json()
        assert body['success'] is True
        res = body['result']
        assert res['predicted_health'] == 'healthy'
        assert res['predicted_crop'] == 'coffee'
        assert res['detected_colors'] == []
        assert res['predicted_pattern'] == 'none'
        assert res['severity'] == 'None'
        assert res['severity_pct'] == 0.0
        assert res['predicted_disease'] == 'No Disease Detected'

    def test_response_contract_keys(self, logged_in_client, healthy_image_path):
        res = _upload(logged_in_client, healthy_image_path).get_json()['result']
        required = {
            'predicted_crop', 'crop_confidence', 'predicted_health',
            'health_confidence', 'predicted_pattern', 'detected_colors',
            'severity', 'severity_pct', 'predicted_disease', 'description',
            'treatment', 'match_score', 'recommendation',
        }
        assert required.issubset(res.keys())
        # new evidence fields
        assert 'pattern_confidence' in res
        assert 'color_evidence' in res
        assert 'crop_ood' in res

    def test_unsupported_crop_rejected(self, logged_in_client, app, healthy_image_path):
        # Force an ambiguous crop prediction (low MSP).
        app.config['CROP_MODEL'] = MockCropModelWithFeatures(logits=(0.1, -0.1))
        r = _upload(logged_in_client, healthy_image_path)
        body = r.get_json()
        assert body['success'] is False
        assert body['reason'] == 'unsupported_crop'

    def test_unhealthy_yellow_leaf_pattern(self, logged_in_client, app, yellow_image_path):
        app.config['HEALTH_MODEL'] = MockHealthModel(logits=(-6.0, 6.0))  # unhealthy
        r = _upload(logged_in_client, yellow_image_path)
        res = r.get_json()['result']
        assert res['predicted_health'] == 'unhealthy'
        assert res['predicted_pattern'] == 'yellowing'
        assert 'yellow' in res['detected_colors']
        assert res['predicted_disease'] != 'Coffee Sooty Mold'
        assert res['severity'] in ('Mild', 'Moderate', 'Severe')

    def test_prediction_logged_to_db(self, logged_in_client, app, healthy_image_path):
        _upload(logged_in_client, healthy_image_path)
        conn = sqlite3.connect(app.config['DATABASE_PATH'] if 'DATABASE_PATH' in app.config
                               else __import__('database').DATABASE_PATH)
        rows = conn.execute('SELECT predicted_crop, predicted_health, predicted_pattern, '
                            'predicted_disease, severity FROM prediction_logs').fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == 'coffee'
        assert rows[0][1] == 'healthy'
        assert rows[0][2] == 'none'

    def test_recommendation_present(self, logged_in_client, healthy_image_path):
        res = _upload(logged_in_client, healthy_image_path).get_json()['result']
        # No GROQ key in tests -> graceful fallback message, never a crash.
        assert isinstance(res['recommendation'], str)
        assert len(res['recommendation']) > 0
