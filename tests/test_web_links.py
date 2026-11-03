"""Sources can open in a browser without dispatching file or active schemes."""

import pytest

from polyrmc.gui.links import open_web_url, safe_web_url


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "file:///C:/Windows/test", "data:text/html,hi",
    "https://name:password@example.org", "https://example.org:bad/",
    "https://example.org:70000/", "https://exa mple.org", "https://example.org/\nmalicious",
])
def test_invalid_source_links_never_reach_the_browser(url, monkeypatch):
    opened = []
    monkeypatch.setattr("polyrmc.gui.links.webbrowser.open_new_tab", opened.append)
    assert safe_web_url(url) is None
    assert open_web_url(url) is False
    assert opened == []


def test_valid_source_opens_its_exact_url(monkeypatch):
    opened = []
    monkeypatch.setattr("polyrmc.gui.links.webbrowser.open_new_tab", lambda url: opened.append(url) or True)
    url = "https://doi.org/10.1234/example?a=1#figure"
    assert open_web_url(url) is True
    assert opened == [url]
