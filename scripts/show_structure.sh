#!/bin/bash
# Show clean project structure

echo "============================================"
echo "Lift-Splat-Shoot SimBEV Project Structure"
echo "============================================"
echo ""

tree -L 2 -I '__pycache__|*.pyc|.git|wandb|*.pt|runs' --dirsfirst \
    -I 'debug_outputs|*.png|*.jpg' \
    --charset ascii

echo ""
echo "============================================"
echo "Key Files:"
echo "============================================"
echo "Training:     train_simbev.py"
echo "Config:       configs/simbev_small.sh"
echo "Data Loader:  src/data_simbev.py"
echo "Model:        src/models.py"
echo ""
echo "Debug Tools:  debug/"
echo "Docs:         docs/"
echo "Scripts:      scripts/"
echo ""
echo "Quick Start:  QUICKSTART.md"
echo "README:       README.md"
echo "============================================"
