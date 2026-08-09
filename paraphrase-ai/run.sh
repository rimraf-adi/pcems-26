#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "✨ Starting Paraphrase AI Studio..."
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m streamlit run app.py
