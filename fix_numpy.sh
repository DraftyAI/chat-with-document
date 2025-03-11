#!/bin/bash

echo "Fixing NumPy compatibility issue with FAISS..."

# Uninstall current numpy
pip uninstall -y numpy

# Install numpy 1.x
pip install 'numpy>=1.21.0,<2.0.0'

# Reinstall faiss-cpu
pip uninstall -y faiss-cpu
pip install 'faiss-cpu>=1.7.0,<1.8.0'

echo "Dependencies reinstalled. Try running the app again."
