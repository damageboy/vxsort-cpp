#!/bin/bash
hogs=$(pgrep -if "(typora|firefox|chrome|chromium-browser|rider|pycharm|resharper|msbuild|telegram|clion|discord|slack)")

resume() {
  echo Resuming "$(echo "$hogs" | wc -w)" procs after running bench
  [[ -z "$hogs" ]] || echo "$hogs" | xargs kill -CONT 
}

trap 'resume' SIGINT

echo Suspending "$(echo "$hogs" | wc -w)" procs before running bench
[[ -z "$hogs" ]] || echo "$hogs" | xargs kill -STOP

SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

"$SCRIPT_DIR"/gcsort_bench --benchmark_counters_tabular "$@"
trap '' SIGINT
resume
