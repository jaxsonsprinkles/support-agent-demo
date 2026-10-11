import pytest


@pytest.fixture(autouse=True)
def _no_flake(monkeypatch):
    """Existing tests assume deterministic tools; flaky-tool tests opt back in with DEMO_FLAKE=1."""
    monkeypatch.setenv("DEMO_FLAKE", "0")
