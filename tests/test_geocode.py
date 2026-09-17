"""Place search (`core.geocode`): the geocoder behind the map's search bar.

These tests never touch the network. Nominatim is stubbed at `urlopen`, which
is the only way in or out of the module, so what is under test is exactly what
the UI depends on:

* nothing is sent for a query too short to be useful (policy: don't load the
  service for noise);
* the request identifies itself (Nominatim's usage policy requires a
  descriptive User-Agent) and is rate limited to <= 1 request/second;
* results become `Place` objects with real coordinates, and impossible
  coordinates are dropped rather than rounded into a plausible lie;
* "no match", a network failure and a malformed response are all reported the
  same honest way: an empty list, never an exception and never a guess.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from core.geocode import (
    ATTRIBUTION,
    MIN_QUERY_CHARS,
    NOMINATIM_URL,
    USER_AGENT,
    Place,
    geocode,
)


# ------------------------------------------------------------------ helpers
class _FakeResponse:
    """Stand-in for the object `urlopen` returns."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _Recorder:
    """Captures the request `geocode` builds, and replays a canned payload."""

    def __init__(self, payload=None, exc=None):
        self.payload = [] if payload is None else payload
        self.exc = exc
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.payload)


@pytest.fixture
def stub(monkeypatch):
    """Patch `urlopen` inside core.geocode and return the recorder."""
    import core.geocode as module

    recorder = _Recorder(payload=[])
    monkeypatch.setattr(module.urllib.request, "urlopen", recorder)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    return recorder


def _item(name="Pune, Maharashtra, India", lat="18.5204", lon="73.8567",
          bbox=None):
    item = {"display_name": name, "lat": lat, "lon": lon}
    if bbox is not None:
        item["boundingbox"] = bbox
    return item


# -------------------------------------------------------------------- tests
def test_a_too_short_query_is_never_sent(stub):
    """Noise must not become load on a free public service."""
    assert geocode("ab") == []
    assert geocode("   ") == []
    assert geocode("") == []
    assert geocode(None) == []
    assert stub.requests == [], "a short query must not reach the geocoder"


def test_the_minimum_length_is_a_named_policy_constant(stub):
    assert MIN_QUERY_CHARS == 3
    geocode("Pune")
    assert len(stub.requests) == 1, "a four-character query is sent"


def test_the_request_identifies_itself(stub):
    """Nominatim's usage policy: a descriptive User-Agent, or no service."""
    stub.payload = []
    geocode("Pune")
    assert len(stub.requests) == 1
    request, timeout = stub.requests[0]
    agent = request.headers.get("User-agent") or request.headers.get("User-Agent")
    assert agent == USER_AGENT
    assert "satquery" in agent.lower()
    assert request.full_url.startswith(NOMINATIM_URL)
    assert "format=jsonv2" in request.full_url
    assert "limit=5" in request.full_url
    assert timeout is not None and timeout > 0


def test_results_become_places_with_real_coordinates(stub):
    stub.payload = [_item(bbox=["18.4", "18.6", "73.7", "73.9"])]
    places = geocode("Pune")
    assert len(places) == 1
    place = places[0]
    assert isinstance(place, Place)
    assert place.name == "Pune, Maharashtra, India"
    assert abs(place.lat - 18.5204) < 1e-9
    assert abs(place.lon - 73.8567) < 1e-9
    assert place.bbox == [18.4, 18.6, 73.7, 73.9]
    zoom = place.zoom_for_bbox
    assert zoom is not None and 2 <= zoom <= 18


def test_a_place_without_a_bbox_falls_back_to_a_sensible_zoom():
    place = Place("Somewhere", 12.0, 34.0, bbox=None)
    assert place.zoom_for_bbox is None
    assert "12.0000" in place.label()


def test_an_impossible_coordinate_is_dropped_not_rounded(stub):
    """91 degrees north is not a place, and rounding it would invent one."""
    stub.payload = [
        _item(name="Impossible", lat="91.0", lon="0.0"),
        _item(name="Real", lat="27.9881", lon="86.9250"),
        _item(name="Bad longitude", lat="10.0", lon="200.0"),
    ]
    places = geocode("anything long enough")
    assert [p.name for p in places] == ["Real"]


def test_unparseable_items_are_skipped(stub):
    stub.payload = [
        {"display_name": "no lat/lon"},
        {"display_name": "nan", "lat": "not-a-number", "lon": "12.0"},
        _item(),
    ]
    places = geocode("Pune")
    assert len(places) == 1
    assert places[0].name == "Pune, Maharashtra, India"


def test_no_match_is_empty_not_an_error(stub):
    stub.payload = []
    assert geocode("definitely not a real place name") == []


def test_a_network_failure_is_reported_as_no_match(stub):
    """The UI shows 'no matches'; it must never surface a geocoder error."""
    stub.exc = urllib.error.URLError("connection refused")
    assert geocode("Pune") == []


def test_a_malformed_response_is_reported_as_no_match(stub, monkeypatch):
    import core.geocode as module

    class _Broken:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):                 # not JSON at all
            return b"<html>502 Bad Gateway</html>"

    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda request, timeout=None: _Broken())
    assert geocode("Pune") == []


def test_the_rate_limit_is_honoured(monkeypatch):
    """Nominatim: at most one request per second."""
    import core.geocode as module

    monkeypatch.setitem(module._last_call, "t", 0.0)   # a cold start
    slept = []
    monkeypatch.setattr(module.time, "sleep", lambda s: slept.append(s))
    recorder = _Recorder(payload=[])
    monkeypatch.setattr(module.urllib.request, "urlopen", recorder)

    geocode("Pune")
    geocode("Mumbai")
    geocode("New Delhi")

    assert len(recorder.requests) == 3
    assert len(slept) == 2, "each request after the first must wait its turn"
    assert all(0.0 < s <= 1.0 for s in slept)


def test_the_geocoder_is_credited_in_the_ui():
    """Attribution is a licence condition of using Nominatim at all."""
    assert "OpenStreetMap" in ATTRIBUTION
    assert "Nominatim" in ATTRIBUTION


def test_search_results_are_bounded():
    """A limit is passed through, so one search cannot ask for everything."""
    import inspect

    source = inspect.getsource(geocode)
    assert "limit" in source
