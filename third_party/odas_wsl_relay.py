#!/usr/bin/env python3
"""
ODAS WSL-side socket relay (bridged-networking workaround).

Why this exists
---------------
WSL is in *bridged* networking mode on a corporate LAN. The Hyper-V external
vSwitch does NOT hairpin VM->host frames, so:

    host  ->  WSL   :  WORKS  (TCP + ICMP)
    WSL   ->  host  :  BLOCKED (TCP + ICMP), even with the firewall wide open.

`odaslive` (running here in WSL) is always a TCP *client*: it connect()s out for
both its raw audio source and its SSL/SST sinks. If those servers live on Windows
it would have to do WSL->host, which is dead. So instead we keep every server on
the WSL side (odaslive connects to 127.0.0.1) and let the *Windows* side connect
INTO WSL (host->WSL, the working direction) to push audio in and pull results out.

Pipes (each forwards bytes one way, in -> out):

  audio :  in  0.0.0.0:5000   (Windows pushes 6-ch int16 PCM)
           out 127.0.0.1:10030 (odaslive raw source connects here)

  ssl   :  in  127.0.0.1:19101 (odaslive SSL sink connects here, sends JSON)
           out 0.0.0.0:19001   (Windows puller connects, forwards to ODAS Studio)

  sst   :  in  127.0.0.1:19105 (odaslive SST sink connects here, sends JSON)
           out 0.0.0.0:19005   (Windows puller connects, forwards to ODAS Studio)

NOTE: the odaslive-facing in-ports are 19101/19105, NOT 9001/9005. WSL
localhostForwarding mirrors WSL 127.0.0.1 listeners onto Windows 127.0.0.1, so
binding 9001/9005 here would shadow ODAS Studio's own Windows ports and loop the
data back into WSL. Unique ports avoid that collision.

Each pipe waits until BOTH ends are connected, then streams in->out until either
side closes, then re-accepts. Run this in WSL before starting odaslive.

Usage (in WSL):
  python3 odas_wsl_relay.py
"""

import socket
import threading

# (name, in_host, in_port, out_host, out_port)
PIPES = [
    ("audio", "0.0.0.0", 5000, "127.0.0.1", 10030),
    ("ssl", "127.0.0.1", 19101, "0.0.0.0", 19001),
    ("sst", "127.0.0.1", 19105, "0.0.0.0", 19005),
]


def make_listener(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


def accept_into(listener, holder, key, name):
    conn, addr = listener.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    holder[key] = conn
    print(f"[{name}] {key} connected from {addr[0]}:{addr[1]}", flush=True)


def run_pipe(name, in_host, in_port, out_host, out_port):
    in_l = make_listener(in_host, in_port)
    out_l = make_listener(out_host, out_port)
    print(f"[{name}] listening  in={in_host}:{in_port}  out={out_host}:{out_port}",
          flush=True)
    while True:
        holder = {}
        ti = threading.Thread(target=accept_into, args=(in_l, holder, "in", name))
        to = threading.Thread(target=accept_into, args=(out_l, holder, "out", name))
        ti.start()
        to.start()
        ti.join()
        to.join()
        src, dst = holder["in"], holder["out"]
        print(f"[{name}] both ends up -> streaming", flush=True)
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError as exc:
            print(f"[{name}] pipe error: {exc}", flush=True)
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass
        print(f"[{name}] reset; waiting for reconnection", flush=True)


def main():
    threads = []
    for cfg in PIPES:
        t = threading.Thread(target=run_pipe, args=cfg, daemon=True)
        t.start()
        threads.append(t)
    print("ODAS WSL relay running. Ctrl-C to stop.", flush=True)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nRelay stopped.")


if __name__ == "__main__":
    main()
