#!/usr/bin/env python3
"""
Windows-side ODAS bridge for bridged WSL networking.

Pairs with third_party/odas_wsl_relay.py (running in WSL). Because only the
host->WSL direction works on this bridged corporate LAN, every connection here
is *outbound* from Windows INTO WSL:

  * audio push : capture the ReSpeaker (6-ch int16 16 kHz) and stream PCM to the
                 relay's audio-in port (WSL 5000). odaslive pulls it locally.

  * sink pull  : connect to the relay's SSL/SST out ports (WSL 19001/19005),
                 read odaslive's JSON results, and forward them to ODAS Studio
                 (listening locally on Windows at 9001/9005).

Startup order:
  1. ODAS Studio (Windows)            -> listens 9001 (SSL), 9005 (SST)
  2. odas_wsl_relay.py (WSL)          -> listens 5000/10030/9001/9005/19001/19005
  3. THIS script (Windows)
  4. odaslive (WSL) with odas_respeaker_socket.cfg (all sockets -> 127.0.0.1)

Usage:
  python odas_win_bridge.py                     # auto-pick ReSpeaker, wsl=192.168.1.50
  python odas_win_bridge.py --wsl-ip 192.168.1.50
  python odas_win_bridge.py --list              # list input devices and exit
"""

import argparse
import socket
import sys
import threading
import time

import pyaudio

TARGET_NAME = "ReSpeaker"


def list_devices(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}  in_ch={info['maxInputChannels']} "
                  f"rate={int(info['defaultSampleRate'])}")


def find_respeaker(pa, want_channels):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if TARGET_NAME in info["name"] and info["maxInputChannels"] >= want_channels:
            return i, info
    return None, None


def connect_retry(host, port, label, stop, retry=2.0):
    """Block until a TCP connection to host:port succeeds (or stop is set)."""
    while not stop.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((host, port))
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[{label}] connected to {host}:{port}", flush=True)
            return s
        except OSError as exc:
            print(f"[{label}] waiting for {host}:{port} ({exc})", flush=True)
            time.sleep(retry)
    return None


def audio_push(args, dev_index, stop):
    pa = pyaudio.PyAudio()
    while not stop.is_set():
        sock = connect_retry(args.wsl_ip, args.audio_port, "audio", stop)
        if sock is None:
            break
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=args.channels,
            rate=args.rate,
            input=True,
            input_device_index=dev_index,
            frames_per_buffer=args.frames,
        )
        print(f"[audio] streaming {args.rate} Hz x {args.channels} ch -> "
              f"{args.wsl_ip}:{args.audio_port}", flush=True)
        try:
            while not stop.is_set():
                data = stream.read(args.frames, exception_on_overflow=False)
                sock.sendall(data)
        except OSError as exc:
            print(f"[audio] disconnected: {exc} -- reconnecting", flush=True)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
    pa.terminate()


def sink_forward(label, args, pull_port, studio_port, stop):
    while not stop.is_set():
        pull = connect_retry(args.wsl_ip, pull_port, f"{label}-pull", stop)
        if pull is None:
            break
        studio = connect_retry(args.studio_host, studio_port, f"{label}-studio", stop)
        if studio is None:
            pull.close()
            break
        print(f"[{label}] forwarding WSL {args.wsl_ip}:{pull_port} -> "
              f"Studio {args.studio_host}:{studio_port}", flush=True)
        try:
            while not stop.is_set():
                data = pull.recv(65536)
                if not data:
                    break
                studio.sendall(data)
        except OSError as exc:
            print(f"[{label}] disconnected: {exc} -- reconnecting", flush=True)
        finally:
            for s in (pull, studio):
                try:
                    s.close()
                except OSError:
                    pass


def main():
    p = argparse.ArgumentParser(description="Windows ODAS bridge (push audio, pull sinks)")
    p.add_argument("--wsl-ip", default="192.168.1.50", help="WSL eth0 IP (relay host).")
    p.add_argument("--audio-port", type=int, default=5000, help="Relay audio-in port.")
    p.add_argument("--ssl-pull-port", type=int, default=19001, help="Relay SSL out port.")
    p.add_argument("--sst-pull-port", type=int, default=19005, help="Relay SST out port.")
    p.add_argument("--studio-host", default="127.0.0.1", help="ODAS Studio host.")
    p.add_argument("--studio-ssl", type=int, default=9001, help="Studio SSL listen port.")
    p.add_argument("--studio-sst", type=int, default=9005, help="Studio SST listen port.")
    p.add_argument("--rate", type=int, default=16000, help="Sample rate (match cfg).")
    p.add_argument("--channels", type=int, default=6, help="Channels (match cfg).")
    p.add_argument("--device", type=int, default=None, help="Input device index.")
    p.add_argument("--frames", type=int, default=1024, help="Frames per read.")
    p.add_argument("--no-sinks", action="store_true", help="Only push audio.")
    p.add_argument("--list", action="store_true", help="List input devices and exit.")
    args = p.parse_args()

    pa = pyaudio.PyAudio()
    if args.list:
        list_devices(pa)
        pa.terminate()
        return
    if args.device is not None:
        dev_index = args.device
        info = pa.get_device_info_by_index(dev_index)
    else:
        dev_index, info = find_respeaker(pa, args.channels)
        if dev_index is None:
            print("ERROR: ReSpeaker input device not found. Devices:")
            list_devices(pa)
            pa.terminate()
            sys.exit(1)
    print(f"Capture device : [{dev_index}] {info['name']} "
          f"(max in_ch={info['maxInputChannels']})")
    pa.terminate()

    stop = threading.Event()
    threads = [threading.Thread(target=audio_push, args=(args, dev_index, stop),
                                daemon=True)]
    if not args.no_sinks:
        threads.append(threading.Thread(
            target=sink_forward,
            args=("ssl", args, args.ssl_pull_port, args.studio_ssl, stop),
            daemon=True))
        threads.append(threading.Thread(
            target=sink_forward,
            args=("sst", args, args.sst_pull_port, args.studio_sst, stop),
            daemon=True))
    for t in threads:
        t.start()
    print("Windows ODAS bridge running. Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop.set()


if __name__ == "__main__":
    main()
