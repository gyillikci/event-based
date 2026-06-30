#!/usr/bin/env bash
# Test WSL -> Windows TCP reachability for the ODAS pipeline ports.
HOST="192.168.1.100"
echo "host gateway / route:"
ip route | head -5
echo "default win host (resolv): $(grep nameserver /etc/resolv.conf 2>/dev/null | awk '{print $2}')"
for port in 10030 9001 9005 10000 10010; do
  if timeout 3 bash -c ">/dev/tcp/$HOST/$port" 2>/dev/null; then
    echo "  $HOST:$port  OPEN"
  else
    echo "  $HOST:$port  CLOSED/blocked"
  fi
done
