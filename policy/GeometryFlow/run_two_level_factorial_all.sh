#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  for label in point128 pointcap148 local_relation local_flow; do
    bash policy/GeometryFlow/run_two_level_factorial_seed.sh "${seed}" "${label}"
  done
done

echo "completed all missing two-level factorial training conditions"
