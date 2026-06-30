#!/usr/bin/env bash
echo "=== interface source files ==="
find ~/odas/src -path '*interface*' -name '*.c' 2>/dev/null
echo "=== connect/bind/listen in src interfaces ==="
grep -rn -e 'connect(' -e 'listen(' -e 'bind(' ~/odas/src 2>/dev/null | grep -i sock | head -30
echo "=== src_hops socket usage ==="
grep -rn -e 'socket' -e 'connect' ~/odas/src/source/src_hops.c 2>/dev/null | head -30
