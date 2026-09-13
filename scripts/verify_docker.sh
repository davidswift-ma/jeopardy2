#!/usr/bin/env bash
# Verify the Docker deployment the way someone else will experience it.
#
# Run from the repo root:   ./scripts/verify_docker.sh
#
# Every check prints PASS or FAIL and the script keeps going, so one run gives
# you the whole picture instead of stopping at the first problem. Exit code is
# non-zero if anything failed.

set -uo pipefail

PORT="${PORT:-8000}"
IMAGE="jeopardy2:verify"
PASS=0
FAIL=0
SKIP=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); }
head2(){ printf '\n\033[1m%s\033[0m\n' "$1"; }

cleanup() {
  head2 "Cleanup"
  docker compose down --remove-orphans >/dev/null 2>&1 && echo "  compose down"
  docker image rm -f "$IMAGE" jeopardy2:amd64 >/dev/null 2>&1
  [ -f .env.verify-backup ] && mv .env.verify-backup .env && echo "  restored .env"
  return 0
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
head2 "1. Toolchain"
# ---------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  ok "docker present ($(docker --version))"
else
  bad "docker not found -- install Docker Desktop and re-run"
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  ok "docker compose present ($(docker compose version --short 2>/dev/null))"
else
  bad "docker compose (v2 plugin) not found"
fi

if docker info >/dev/null 2>&1; then
  ok "docker daemon is running"
else
  bad "docker daemon not running -- start Docker Desktop"
  exit 1
fi

# ---------------------------------------------------------------------------
head2 "2. Compose file is valid"
# ---------------------------------------------------------------------------
if docker compose config >/dev/null 2>&1; then
  ok "docker compose config parses"
else
  bad "docker compose config failed:"
  docker compose config 2>&1 | sed 's/^/        /'
fi

# ---------------------------------------------------------------------------
head2 "3. Image builds (native architecture)"
# ---------------------------------------------------------------------------
if docker build -t "$IMAGE" . >/tmp/j2-build.log 2>&1; then
  ok "docker build succeeded"
  SIZE=$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null)
  [ -n "${SIZE:-}" ] && echo "        image size: $((SIZE/1024/1024)) MB"
else
  bad "docker build FAILED -- last 30 lines:"
  tail -30 /tmp/j2-build.log | sed 's/^/        /'
  echo "        (full log: /tmp/j2-build.log)"
fi

# ---------------------------------------------------------------------------
head2 "4. Builds for the other CPU architecture"
# ---------------------------------------------------------------------------
# Your Mac is arm64. If your professor is on an Intel Mac or Windows/Linux
# x86, the image has to build there too -- this is the single most likely way
# the handoff breaks.
NATIVE_ARCH=$(docker info --format '{{.Architecture}}' 2>/dev/null)
echo "        this machine: ${NATIVE_ARCH:-unknown}"
if docker buildx version >/dev/null 2>&1; then
  if docker buildx build --platform linux/amd64 -t jeopardy2:amd64 --load . \
       >/tmp/j2-amd64.log 2>&1; then
    ok "builds for linux/amd64 (Intel/Windows professor is covered)"
  else
    bad "linux/amd64 build FAILED -- last 20 lines:"
    tail -20 /tmp/j2-amd64.log | sed 's/^/        /'
  fi
else
  skip "buildx unavailable; cannot test x86 build"
fi

# ---------------------------------------------------------------------------
head2 "5. Fresh-clone case: starts with NO .env file"
# ---------------------------------------------------------------------------
# .env is gitignored, so this is exactly what your professor gets on clone.
# It must start and explain itself, not crash.
[ -f .env ] && mv .env .env.verify-backup

if docker compose up -d --build >/tmp/j2-up-noenv.log 2>&1; then
  ok "compose up succeeded without a .env file"

  READY=""
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
      READY=1; break
    fi
    sleep 1
  done

  if [ -n "$READY" ]; then
    ok "GET /health responds 200"
  else
    bad "/health never came up -- container logs:"
    docker compose logs --tail 30 2>&1 | sed 's/^/        /'
  fi

  if [ -n "$READY" ]; then
    CONFIG=$(curl -fsS --max-time 5 "http://localhost:$PORT/jeopardy2/config")
    if echo "$CONFIG" | grep -q '"openai": false'; then
      ok "/jeopardy2/config correctly reports the missing keys"
    else
      bad "expected credentials_present.openai=false; got: $CONFIG"
    fi

    # The no-5xx promise, with nothing configured at all.
    CODE=$(curl -s -o /tmp/j2-degraded.json -w '%{http_code}' --max-time 20 \
      -X POST "http://localhost:$PORT/jeopardy2" \
      -H 'Content-Type: application/json' \
      -d '{"question":"Explain AI simply."}')
    if [ "$CODE" = "200" ] && grep -q '"status":"degraded"' /tmp/j2-degraded.json; then
      ok "keyless request returns 200 + status:degraded (no 5xx)"
    else
      bad "expected 200/degraded, got HTTP $CODE: $(cat /tmp/j2-degraded.json)"
    fi

    if curl -fsS --max-time 5 "http://localhost:$PORT/" | grep -qi jeopardy; then
      ok "browser UI is served at /"
    else
      bad "UI did not render at /"
    fi
  fi

  # Security and ops properties of the running container.
  WHO=$(docker compose exec -T agent whoami 2>/dev/null | tr -d '\r\n')
  if [ "$WHO" = "app" ]; then
    ok "container runs as non-root (user: app)"
  else
    bad "expected to run as 'app', got '${WHO:-<none>}'"
  fi

  if docker compose exec -T agent test -f data/jeopardy_sample.tsv 2>/dev/null; then
    ROWS=$(docker compose exec -T agent sh -c 'wc -l < data/jeopardy_sample.tsv' 2>/dev/null | tr -d ' \r\n')
    ok "dataset sample present in image (${ROWS:-?} lines)"
  else
    bad "data/jeopardy_sample.tsv missing inside the container"
  fi

  if docker compose exec -T agent sh -c 'env | grep -i "api_key"' 2>/dev/null | grep -q .; then
    bad "an API key is set inside the container when .env is absent -- possible leak into the image"
  else
    ok "no API keys baked into the image"
  fi

  sleep 25  # let the HEALTHCHECK report at least once
  HS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
        "$(docker compose ps -q agent)" 2>/dev/null)
  case "$HS" in
    healthy)  ok "container HEALTHCHECK reports healthy" ;;
    starting) skip "healthcheck still 'starting' (give it longer)" ;;
    *)        bad "healthcheck status: ${HS:-unknown}" ;;
  esac

  docker compose down >/dev/null 2>&1
else
  bad "compose up FAILED without .env -- last 30 lines:"
  tail -30 /tmp/j2-up-noenv.log | sed 's/^/        /'
fi

[ -f .env.verify-backup ] && mv .env.verify-backup .env

# ---------------------------------------------------------------------------
head2 "6. Real end-to-end answer (needs your keys in .env)"
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  skip "no .env present; skipping the live-provider test"
elif ! grep -qE '^(OPENAI|ANTHROPIC)_API_KEY=[^[:space:]]+$' .env 2>/dev/null \
   || ! grep -vE '^(OPENAI_API_KEY=sk-\.\.\.|ANTHROPIC_API_KEY=sk-ant-\.\.\.)$' .env 2>/dev/null \
        | grep -qE '^(OPENAI|ANTHROPIC)_API_KEY=[^[:space:]]+$'; then
  # Also treats the unedited .env.example placeholders as "no key", so a
  # forgotten edit reads as SKIP rather than a mysterious 401 FAIL.
  skip ".env has no real API key (still the placeholder?); skipping live test"
else
  if docker compose up -d >/tmp/j2-up-env.log 2>&1; then
    for _ in $(seq 1 30); do
      curl -fsS --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1 && break
      sleep 1
    done

    CODE=$(curl -s -o /tmp/j2-live.json -w '%{http_code}' --max-time 120 \
      -X POST "http://localhost:$PORT/jeopardy2" \
      -H 'Content-Type: application/json' \
      -d '{"question":"Explain AI in one or two sentences that my grandfather could understand."}')

    if [ "$CODE" = "200" ] && grep -q '"status":"ok"' /tmp/j2-live.json; then
      ok "live question answered by a real provider"
      echo "        served by: $(grep -o '"served_by_model":"[^"]*"' /tmp/j2-live.json)"
      echo "        answer:    $(sed -n 's/.*"answer":{"answer":"\([^"]\{0,110\}\).*/\1/p' /tmp/j2-live.json)..."
    else
      bad "live request returned HTTP $CODE / not ok. Body:"
      sed 's/^/        /' /tmp/j2-live.json
      echo "        A 404 from a provider usually means a stale model ID in .env"
      echo "        (OPENAI_MODEL / ANTHROPIC_MODEL), not a code bug."
    fi

    # The fallback chain, over the real deployment.
    if curl -fsS --max-time 120 \
         "http://localhost:$PORT/jeopardy2/stream?question=hi&force_fail=openai" \
         2>/dev/null | grep -q 'falling_back'; then
      ok "forced-failure fallback to Claude works in the container"
    else
      bad "force_fail=openai did not produce a falling_back event"
    fi

    docker compose down >/dev/null 2>&1
  else
    bad "compose up FAILED with .env -- last 30 lines:"
    tail -30 /tmp/j2-up-env.log | sed 's/^/        /'
  fi
fi

# ---------------------------------------------------------------------------
head2 "Summary"
# ---------------------------------------------------------------------------
printf '  %d passed, %d failed, %d skipped\n\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ] || exit 1
