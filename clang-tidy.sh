#!/bin/bash

if [ $# -ne 1 ] ; then
	echo "Usage $0 <build_folder>" >&2
	exit 1
fi

BUILD_DIR=$1
NPROC=${NPROC_CI:-$(nproc)}

for candidate in run-clang-tidy-12 run-clang-tidy-11 run-clang-tidy-10 run-clang-tidy ; do
	if command -v $candidate >/dev/null ; then
		echo "Using '$candidate' to execute clang-tidy in parallel"
		_RUN_CLANG_TIDY=$candidate
		break
	fi
done

if [ -z ${_RUN_CLANG_TIDY} ] >/dev/null ; then
	echo "run-clang-tidy not found in PATH" >&2
	exit 1
fi

ORIGINAL_COMPILE_COMMANDS="$BUILD_DIR"/compile_commands.json

CXX_PROJECT_DIR="$(mktemp -d --suffix='-clang-tidy-vxsort')"
jq 'map(select( (.["file"] | contains("/googletest") | not) and (.["file"] | contains("/googlebenchmark") | not) and (.["file"] | contains("/cpu_features") | not) ))' \
        < "$ORIGINAL_COMPILE_COMMANDS"          \
        > "$CXX_PROJECT_DIR"/compile_commands.json || exit 1
exec "${_RUN_CLANG_TIDY}" \
        -j "${NPROC}" \
        -p "$CXX_PROJECT_DIR" \
        -config="$(cat .clang-tidy.cxx.yaml)"

