import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 5000))
s.listen(1)
print("WSL listener ready on 0.0.0.0:5000", flush=True)
s.settimeout(20)
try:
    c, a = s.accept()
    print("conn from", a, flush=True)
    c.sendall(b"HELLO_FROM_WSL")
    c.close()
    print("sent banner, ok", flush=True)
except Exception as e:
    print("listener error:", e, flush=True)
