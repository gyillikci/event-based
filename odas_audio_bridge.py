#!/usr/bin/env python3
"""
Windows -> ODAS raw-audio bridge for the ReSpeaker USB 4 Mic Array.

ODAS Studio only visualizes; the DOA math is done by `odaslive` (Linux/WSL).
`odaslive` reads raw microphone audio, but in bridged WSL networking it can't
grab the USB mic directly. This bridge solves that:

  ReSpeaker (Windows, PyAudio, 6 ch int16 16 kHz)
        |  interleaved int16 PCM over TCP
        v
  odaslive (WSL)  --connects in as a client-->  this bridge (TCP server)

odaslive's raw `socket` source uses connect() (it is the CLIENT) and reads
hopSize*nChannels*2 bytes per hop via MSG_WAITALL, interleaved int16 LE. We just
stream the device's native interleaved int16 frames continuously; chunking is
irrelevant because the reader reassembles with MSG_WAITALL.

Match these to odas_respeaker_socket.cfg:  fS=16000, nChannels=6, nBits=16.

Usage:
  python odas_audio_bridge.py                 # bind 0.0.0.0:10030, auto-pick ReSpeaker
  python odas_audio_bridge.py --port 10030 --channels 6
  python odas_audio_bridge.py --list          # list input devices and exit
"""

import argparse
import socket
import sys

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
        if (TARGET_NAME in info["name"]
                and info["maxInputChannels"] >= want_channels):
            return i, info
    return None, None


def main():
    p = argparse.ArgumentParser(description="ReSpeaker -> ODAS raw audio TCP bridge")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default all).")
    p.add_argument("--port", type=int, default=10030, help="TCP port to serve PCM.")
    p.add_argument("--rate", type=int, default=16000, help="Sample rate (match cfg).")
    p.add_argument("--channels", type=int, default=6, help="Channels (match cfg).")
    p.add_argument("--device", type=int, default=None, help="Input device index.")
    p.add_argument("--frames", type=int, default=1024, help="Frames per read.")
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
    print(f"Format         : {args.rate} Hz, {args.channels} ch, int16 interleaved")
    print(f"Serving on     : {args.host}:{args.port}  (odaslive connects in)")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    try:
        while True:
            print("Waiting for odaslive to connect...")
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"odaslive connected from {addr[0]}:{addr[1]} - streaming.")

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=args.channels,
                rate=args.rate,
                input=True,
                input_device_index=dev_index,
                frames_per_buffer=args.frames,
            )
            try:
                while True:
                    data = stream.read(args.frames, exception_on_overflow=False)
                    conn.sendall(data)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                print("odaslive disconnected.")
            finally:
                stream.stop_stream()
                stream.close()
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\nBridge stopped.")
    finally:
        srv.close()
        pa.terminate()


if __name__ == "__main__":
    main()
