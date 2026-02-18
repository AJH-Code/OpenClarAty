#!/usr/bin/env python3
"""Simple HTTP server for CLaRa Memory Brain Visualizer on port 8400."""

import http.server
import socketserver
import os

PORT = 8400
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # CORS headers for API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[BrainViz] {args[0]}")

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"CLaRa Memory Brain Visualizer")
        print(f"Serving on http://0.0.0.0:{PORT}")
        print(f"Open http://192.168.88.80:{PORT} in your browser")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
