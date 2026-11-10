"""Expose the existing remote control server through the configured SSH proxy."""
import argparse
from pathlib import Path
import select
import socketserver
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import server_exec

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=8768)
p.add_argument('--remote-port', type=int, default=8768)
a = p.parse_args()
client = server_exec.connect()
connection_lock=threading.Lock()

def transport():
    global client
    with connection_lock:
        current=client.get_transport()
        if current is None or not current.is_active():
            client.close()
            client=server_exec.connect()
        return client.get_transport()

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        channel = transport().open_channel('direct-tcpip',
            ('127.0.0.1', a.remote_port), self.client_address)
        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 30)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (channel if source is self.request else self.request).sendall(data)
        finally:
            channel.close()

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

with Server(('127.0.0.1', a.port), Handler) as server:
    print(f'Control server: http://127.0.0.1:{a.port}', flush=True)
    server.serve_forever()
