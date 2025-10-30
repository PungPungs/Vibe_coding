#!/bin/bash
# Build script for SEG-Y 2D Viewer (Rust)

echo "Building SEG-Y 2D Viewer (Rust)..."
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed"
    echo "Install Rust from: https://rustup.rs/"
    exit 1
fi

# Build release version
echo "Building release version..."
cargo build --release

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Executable location: target/release/segy_2d_viewer_rust"
    echo ""
    echo "Run with: cargo run --release"
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi
