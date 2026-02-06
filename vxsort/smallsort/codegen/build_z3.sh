#!/usr/bin/env bash
set -euo pipefail

# Build script for Z3 with Python 3.13t free-threaded support

echo "=================================="
echo "Z3 Build Script for Python 3.13t"
echo "=================================="
echo

# Activate venv if it exists (to use uv-managed Python)
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Detect Python interpreter (prefer uv-managed Python)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found"
    echo "Please install Python 3.13t first:"
    echo "  uv python install 3.13t"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo

# Get detailed Python version
PYTHON_VERSION=$($PYTHON_CMD -VV 2>&1 | head -n 1)
echo "Python version: $PYTHON_VERSION"

# Check if it's Python 3.13+
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [[ "$PYTHON_MAJOR" -ne 3 ]] || [[ "$PYTHON_MINOR" -lt 13 ]]; then
    echo "Error: Python 3.13+ required, found $PYTHON_MAJOR.$PYTHON_MINOR"
    echo "Please install Python 3.13t:"
    echo "  uv python install 3.13t"
    exit 1
fi

# Verify it's the free-threaded build
if [[ ! "$PYTHON_VERSION" =~ "free-threading" ]]; then
    echo "Error: Python does not appear to be a free-threaded build"
    echo "Expected 'free-threading' in version string, got:"
    echo "  $PYTHON_VERSION"
    echo
    echo "Please ensure you're using Python 3.13t (the 't' variant):"
    echo "  uv python install 3.13t"
    echo "  uv venv --python 3.13t"
    exit 1
fi

echo "✓ Free-threaded Python detected"
echo

# Set build directory
BUILD_DIR="${BUILD_DIR:-../z3-build}"
Z3_REPO="${Z3_REPO:-https://github.com/Z3Prover/z3.git}"

echo "Configuration:"
echo "  Build directory: $BUILD_DIR"
echo "  Z3 repository: $Z3_REPO"
echo

# Clone Z3 if not already present
if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning Z3 repository..."
    git clone "$Z3_REPO" "$BUILD_DIR"
else
    echo "Z3 repository already exists, updating..."
    cd "$BUILD_DIR"
    git pull
    cd -
fi

echo
echo "Building Z3 with Python 3.13t bindings..."
cd "$BUILD_DIR"

# Clean previous build
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

# Configure and build
echo "Running mk_make.py with $PYTHON_CMD..."
$PYTHON_CMD scripts/mk_make.py --python

echo "Building..."
cd build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo
echo "Installing z3-solver package..."
make install

echo
echo "=================================="
echo "Build completed successfully!"
echo "=================================="
echo
echo "Z3 has been built and installed for Python 3.13t"
echo "You can now run the demo:"
echo "  cd $(dirname "$0")"
echo "  uv run python main.py"
