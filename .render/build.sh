#!/usr/bin/env bash
echo "Forcing classic pip installation (not poetry)..."
pip install --upgrade pip
pip install -r requirements.txt
