"""Launch SatQuery with the same-origin tile proxy mounted on the app server.

    python serve.py

Streamlit does not expose a public API for custom HTTP routes, but it builds a
Starlette application through `create_starlette_app`, which is imported by name
into `starlette_server` and called at startup. Wrapping that call lets us add one
route -- `/satquery-tiles/<provider>/<z>/<x>/<y>.png` -- to the app's *own*
origin. That matters: the browser can reach the app, so it can reach the tiles,
and it never has to talk to a tile provider that might answer 403.

Everything else is ordinary `streamlit run app.py`; the CLI is invoked unchanged
after the wrapper is installed, so no Streamlit behaviour is modified.
"""

from __future__ import annotations

import sys


def install() -> None:
    import streamlit.web.server.starlette.starlette_server as starlette_server
    import tileserver

    original = starlette_server.create_starlette_app
    if getattr(original, "_satquery_patched", False):
        return

    def create_starlette_app(runtime):  # noqa: ANN001, ANN202
        app = original(runtime)
        try:
            tileserver.install(app)
            print("[satquery] tile proxy mounted at /satquery-tiles", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[satquery] tile proxy NOT mounted: {exc!r}", flush=True)
        return app

    create_starlette_app._satquery_patched = True
    starlette_server.create_starlette_app = create_starlette_app


def main() -> int:
    install()
    from streamlit.web.cli import main as streamlit_cli

    argv = sys.argv[1:]
    if not argv or argv[0] != "run":
        argv = ["run", *argv]
    # Keep the CLI defaults the rest of the project relies on.
    defaults = ["--server.port", "8501", "--server.address", "0.0.0.0"]
    for flag, value in (("--server.port", "8501"), ("--server.address", "0.0.0.0")):
        if flag not in argv:
            argv += [flag, value]
    if "app.py" not in argv:
        argv.insert(1, "app.py")
    sys.argv = ["streamlit", *argv]
    streamlit_cli(prog_name="streamlit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
