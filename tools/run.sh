#!/usr/bin/env bash
# Run zkbot vs CircuitAI headless and print the report.
#   tools/run.sh <tag> <Normal|Hard|Brutal> [games=1] [map='TitanDuel 2.2'] [minutes=6]
# ZKBOT_PY selects the bot interpreter (default python), e.g. ZKBOT_PY=pypy3.
# Blocks until all games finish; output goes to results/<tag>.
set -u
cd "$(dirname "$0")/.."
TAG=$1; DIFF=$2; GAMES=${3:-1}; MAP=${4:-TitanDuel 2.2}; MINUTES=${5:-6}
taskkill //IM spring-headless.exe //F >/dev/null 2>&1 || true
taskkill //IM spring.exe //F >/dev/null 2>&1 || true
sleep 1
rm -rf "results/$TAG"
ALT=""
if [ "$GAMES" -gt 1 ]; then ALT="-AlternateSides"; fi
LIMIT=$(( GAMES * (MINUTES * 60 + 90) + 60 ))
timeout "$LIMIT" powershell -NoProfile -ExecutionPolicy Bypass -Command \
  ".\\zk-startup.ps1 ControllerAI CircuitAI:$DIFF -Headless -Map '$MAP' -NoDownload \
   -TimeoutMinutes $MINUTES -Games $GAMES $ALT -OutDir 'results/$TAG' \
   -Client '${ZKBOT_PY:-python} bots\\zkbot\\main.py {team}'" > "results/$TAG.out" 2>&1
taskkill //IM spring-headless.exe //F >/dev/null 2>&1 || true
python tools/report.py "results/$TAG"
