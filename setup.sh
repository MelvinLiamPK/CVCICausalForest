#!/bin/bash
# ============================================================
# CVCI-CF Project Setup
# Run from: /Users/melvinliam/Documents/Work/Self-Projects/CVCICF
# ============================================================

set -e

PROJECT_DIR="/Users/melvinliam/Documents/Work/Self-Projects/CVCICF"

# Create project directory and navigate
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Initialize git
git init 2>/dev/null || true
git remote add origin https://github.com/MelvinLiamPK/CVCICausalForest.git 2>/dev/null || true

# Create directory structure
mkdir -p src tests experiments results notebooks docs

echo "✓ Project structure created at $PROJECT_DIR"
echo ""
echo "Now copy the project files into the directory, then run:"
echo "  cd $PROJECT_DIR"
echo "  pip install -r requirements.txt"
echo "  python tests/test_sanity.py"
echo ""
echo "To push to GitHub:"
echo "  git add -A"
echo "  git commit -m 'Initial project setup: CVCI with Causal Forests'"
echo "  git branch -M main"
echo "  git push -u origin main"
