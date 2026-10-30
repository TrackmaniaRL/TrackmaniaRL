"""Serve one owned ghost locally, then exit; no directory access or uploads."""
import http.server
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
if sys.platform == "win32":
    path = pathlib.Path("\\\\?\\" + str(path.absolute()))
content = path.read_bytes()
served = False


class Handler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()

    def do_GET(self) -> None:
        global served
        if self.path != "/model.Ghost.Gbx":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
        served = True
        print("MODEL_GHOST1110 served", len(content), flush=True)

    def log_message(self, format: str, *args: object) -> None:
        print(self.command, self.path, format % args, flush=True)


with http.server.HTTPServer(("127.0.0.1", 18710), Handler) as server:
    server.timeout = 1
    deadline = time.monotonic() + 180
    print("MODEL_GHOST1110 local server ready", flush=True)
    while not served and time.monotonic() < deadline:
        server.handle_request()
if not served:
    raise SystemExit("No client request within bound")
