"""Open a trained 3DGS ``.ply`` in the SuperSplat web editor.

SuperSplat (https://superspl.at/editor) can load a splat straight from a URL
via its ``?load=<url>`` parameter -- but that URL has to be reachable by the
browser *and* CORS-enabled. A freshly-exported local ``.ply`` is neither, so
this serves the file's folder over a short-lived ``localhost`` HTTP server
(adding an ``Access-Control-Allow-Origin: *`` header) and points the editor at
it. ``localhost`` is treated as a secure origin, so the https editor is allowed
to fetch the plain-http file.

    python -m raven.view_splat                  # opens <data>/work/splat.ply
    python -m raven.view_splat --ply path.ply
    python -m raven.view_splat --no-browser     # just print the URL

It is also called from :mod:`raven.train_splat` (``--view``) and the cloud
editor's "Open in SuperSplat" button.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

from .paths import Capture, add_data_arg

EDITOR_URL = "https://superspl.at/editor"


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that adds the CORS header SuperSplat's fetch needs."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):  # keep the console quiet
        pass


def _serve(directory: Path) -> tuple[socketserver.TCPServer, int]:
    """Serve ``directory`` on a free localhost port in a daemon thread."""
    handler = functools.partial(_CORSHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)  # port 0 => OS picks
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def supersplat_url(ply_name: str, port: int) -> str:
    """Build the ``superspl.at/editor?load=...`` URL for a served file."""
    file_url = f"http://localhost:{port}/{quote(ply_name)}"
    return f"{EDITOR_URL}?load={quote(file_url, safe='')}"


def open_in_supersplat(ply: str | Path, *, open_browser: bool = True):
    """Serve ``ply`` locally and open it in SuperSplat. Returns ``(httpd, url)``.

    The caller owns the returned server: keep it alive while the editor may
    still fetch the file (the GUI holds it for the process lifetime; the CLI
    blocks until Ctrl-C), then call ``httpd.shutdown()``.
    """
    ply = Path(ply).expanduser().resolve()
    if not ply.exists():
        raise FileNotFoundError(f"no splat to open: {ply}")
    httpd, port = _serve(ply.parent)
    url = supersplat_url(ply.name, port)
    if open_browser:
        webbrowser.open(url)
    return httpd, url


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--ply", default=None,
                    help="splat .ply to open (default <data>/work/splat.ply)")
    ap.add_argument("--no-browser", action="store_true",
                    help="just print the URL; don't launch a browser")
    args = ap.parse_args()

    ply = Path(args.ply) if args.ply else Capture.from_args(args).p("splat.ply")
    httpd, url = open_in_supersplat(ply, open_browser=not args.no_browser)
    print(f"serving {ply.name} for SuperSplat at:\n  {url}\nCtrl-C to stop serving.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
