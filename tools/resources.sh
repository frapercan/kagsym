#!/bin/bash
# Before every launch: what the machine has left. Use it near the limit, never over it.
cores=$(nproc); load=$(cut -d' ' -f1 /proc/loadavg); free_mb=$(free -m | awk 'NR==2{print $7}')
heavy=$(ps -eo pcpu,args | awk '$1 > 20 && /python/ {n++} END{print n+0}')
room=$(python3 -c "print(max(0, int($cores - 1 - $load + 0.5)))")
echo "cores $cores | load $load | heavy python procs $heavy | RAM available ${free_mb} MB | room for ~$room more workers (keep 1 core free)"
