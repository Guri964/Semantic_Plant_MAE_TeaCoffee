"""Knowledge-base matching tests (health consistency + no invented diagnoses)."""

import importlib
import os

import pytest

import config


@pytest.fixture
def kb_db(tmp_path, monkeypatch):
    db_path = os.path.join(str(tmp_path), 'kb.db')
    monkeypatch.setattr(config, 'DATABASE_PATH', db_path, raising=False)
    import database
    importlib.reload(database)
    monkeypatch.setattr(database, 'DATABASE_PATH', db_path, raising=False)
    database.init_db()
    import services.knowledge_base as kb
    importlib.reload(kb)
    monkeypatch.setattr(kb, 'DATABASE_PATH', db_path, raising=False)
    return kb


class TestKnowledgeBase:
    def test_seeded_sixteen_conditions(self, kb_db):
        import sqlite3
        conn = sqlite3.connect(kb_db.DATABASE_PATH)
        n = conn.execute('SELECT COUNT(*) FROM knowledge_base').fetchone()[0]
        conn.close()
        assert n == 16

    def test_healthy_matches_healthy_row(self, kb_db):
        res = kb_db.match_disease('coffee', 'healthy', [])
        assert res['disease'] == 'Healthy Coffee Leaf'
        assert res['match_score'] >= 0.99

    def test_healthy_never_matches_disease_even_with_colors(self, kb_db):
        res = kb_db.match_disease('coffee', 'healthy', ['black', 'gray'])
        assert res['disease'] == 'Healthy Coffee Leaf'

    def test_yellow_evidence_not_sooty_mold(self, kb_db):
        """Issue 1: yellow patches must not map to Sooty Mold (black,gray)."""
        res = kb_db.match_disease('coffee', 'unhealthy', ['yellow'])
        assert res['disease'] != 'Coffee Sooty Mold'
        # yellow supports Rust (brown,yellow) or Chlorosis (yellow,white)
        assert res['disease'] in ('Coffee Leaf Rust', 'Coffee Chlorosis')

    def test_black_gray_evidence_sooty_mold(self, kb_db):
        res = kb_db.match_disease('coffee', 'unhealthy', ['black', 'gray'])
        assert res['disease'] == 'Coffee Sooty Mold'

    def test_unhealthy_no_colors_inconclusive(self, kb_db):
        """No color evidence must not invent a specific disease."""
        res = kb_db.match_disease('coffee', 'unhealthy', [])
        assert res['disease'] == 'Condition Not Identified'
        assert res['pattern'] == 'inconclusive'

    def test_crop_isolation(self, kb_db):
        res = kb_db.match_disease('tea', 'unhealthy', ['white', 'gray'])
        assert res['disease'].startswith('Tea')

    def test_missing_table_graceful(self, tmp_path, monkeypatch):
        import services.knowledge_base as kb
        monkeypatch.setattr(kb, 'DATABASE_PATH',
                            os.path.join(str(tmp_path), 'empty.db'), raising=False)
        res = kb.match_disease('coffee', 'unhealthy', ['brown'])
        assert res['disease'] == 'Condition Not Identified'
