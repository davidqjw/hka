import json
import sys
import os
from io import StringIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import webbrowser
import traceback

from sequential_pipeline import run_pipeline

ROOT = Path(__file__).parent.resolve()
PORT = 8765

print(f"[INFO] ROOT: {ROOT}")
print(f"[INFO] UI_PATH: {ROOT / 'ui'}")
print(f"[INFO] ui.html: {ROOT / 'ui' / 'ui.html'} - exists: {(ROOT / 'ui' / 'ui.html').exists()}")
print(f"[INFO] app.js: {ROOT / 'ui' / 'app.js'} - exists: {(ROOT / 'ui' / 'app.js').exists()}")

if not (ROOT / "ui" / "ui.html").exists():
    print(f"[ERROR] ui.html not found!")
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        sys.stderr.write(f"[server] {format % args}\n")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            path = self.path.split('?')[0].split('#')[0]
            
            print(f"[GET] Raw path: {path}", file=sys.stderr)
            
            if path == '/' or path == '':
                path = '/ui/ui.html'
            
            if path.startswith('/'):
                path = path[1:]
            
            file_path = ROOT / path
            
            print(f"[GET] Request: {self.path} -> {path} -> {file_path}", file=sys.stderr)
            
            if not file_path.exists() or not file_path.is_file():
                print(f"[GET] File not found: {file_path}", file=sys.stderr)
                self.send_error(404, f"File not found: {path}")
                return
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if path.endswith('.html'):
                content_type = 'text/html; charset=utf-8'
            elif path.endswith('.js'):
                content_type = 'application/javascript; charset=utf-8'
            elif path.endswith('.css'):
                content_type = 'text/css; charset=utf-8'
            elif path.endswith('.json'):
                content_type = 'application/json; charset=utf-8'
            else:
                content_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            
            print(f"[GET] Served: {path} ({len(content)} bytes)", file=sys.stderr)
            
        except Exception as e:
            print(f"[GET] Error: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            self.send_error(500, str(e))

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            
            print(f"[POST] /api/chat - {length} bytes", file=sys.stderr)

            # what: parse json
            # what: message is a list contains content and role
            data = json.loads(body.decode("utf-8"))
            messages = data.get("messages", [])
            
            print(f"[POST] Messages count: {len(messages)}", file=sys.stderr)

            # what: get the last "user" message
            query = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = str(msg.get("content", "")).strip()
                    break
            
            print(f"[POST] Query: {repr(query[:100])}", file=sys.stderr)


            # what: get result from sequ_pipeline
            try:
                result = run_pipeline([query] if query else [""])
            except Exception as e:
                print(f"[POST] Pipeline error: {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                raise

            if result is None:
                result_str = f"{self}No output from pipeline"
            else:
                result_str = str(result).strip()
                if not result_str:
                    result_str = f"{self}Wrong to convert result"
            
            print(f"[POST] Result: {repr(result_str[:100])}", file=sys.stderr)


            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response = json.dumps({
                "output_text": result_str,
                "status": "success"
            }, ensure_ascii=False)
            
            self.wfile.write(response.encode("utf-8"))
            print(f"[POST] Response sent ({len(response)} bytes)", file=sys.stderr)

        except json.JSONDecodeError as e:
            print(f"[POST] JSON decode error: {e}", file=sys.stderr)
            self._send_json_error(400, f"Invalid JSON: {e}")
            
        except Exception as e:
            print(f"[POST] Error: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            self._send_json_error(500, str(e))

    def _send_json_error(self, code, message):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            error_response = json.dumps({
                "output_text": f"[Error {code}] {message}",
                "status": "error",
                "error": message
            }, ensure_ascii=False)
            
            self.wfile.write(error_response.encode("utf-8"))
        except:
            pass


# what: run the server
def run():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, Handler)
    
    url = f"http://localhost:{PORT}/ui/ui.html"
    
    print()
    print("=" * 70)
    print("Hokie Knowledge Agent Server")
    print("=" * 70)
    print(f"Server:  http://localhost:{PORT}")
    print(f"UI:      {url}")
    print(f"API:     http://localhost:{PORT}/api/chat")
    print(f"Files:   {ROOT}")
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    # browser
    print(f"Please open the link: {url}\n")
    # try:
    #     webbrowser.open(url)
    #     print(f"Browser opened: {url}\n")
    # except Exception as e:
    #     print(f"Could not open browser: {e}")
    #     print(f"  Please open manually: {url}\n")

    # server
    try:
        print("Server is running...\n")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        httpd.shutdown()
        print("Server stopped\n")


if __name__ == "__main__":
    run()