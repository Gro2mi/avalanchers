import http.server
import ssl
import socket
import os 
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip = s.getsockname()[0]
s.close()

certfile="server-cert.pem"
keyfile="server-key.pem"
if not os.path.exists(certfile) or not os.path.exists(keyfile):
    command = [
        "openssl", "req", "-new", "-x509", "-keyout", keyfile,
        "-out", certfile, "-days", "365", "-nodes",
        "-subj", "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost"
    ]

    print("Creating a self-signed certificate")
    result = subprocess.run(command, capture_output=True, text=True)

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set cache-control headers to disable caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

# Set the server's address and port
server_address = ('', 443)  # Empty string means it will listen on all available interfaces, port 4443

# Create the server
httpd = http.server.HTTPServer(server_address, NoCacheHandler)

# Create an SSLContext
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile=certfile, keyfile=keyfile)

# Wrap the server's socket with the SSL context
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

# Start the server
print("Open in Chrome: https://" + ip + "/index.html?debug=vscode")

httpd.serve_forever()
