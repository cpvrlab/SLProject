from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import os

HOSTNAME = "0.0.0.0"
PORT = 8080


class RequestHandler(BaseHTTPRequestHandler):
    MIME_TYPES = {
        ".html": "text/html",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".data": "application/octet-stream"
    }

    def do_GET(self):
        file_path = self.get_path()

        try:
            file = open(file_path, "rb")
            content = file.read()
            file.close()

            file_ext = file_path[file_path.rindex("."):]
            mime_type = RequestHandler.MIME_TYPES.get(file_ext, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            print(e)
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        file_path = self.get_path()
        self.send_response(200 if os.path.isfile(file_path) else 404)
        self.end_headers()

    def get_path(self):
        url = urllib.parse.urlparse(self.path)
        file_path = url.path
        if file_path == "/":
            file_path = "/app-Demo-SLProject.html"
        if file_path.startswith("/data/"):
            file_path = "/.." + file_path

        return os.curdir + file_path


server = HTTPServer((HOSTNAME, PORT), RequestHandler)

print(f"serving on port {PORT}")
server.serve_forever()
