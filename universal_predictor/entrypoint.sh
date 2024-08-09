#!/bin/sh

# Run the training script
python pyfiles/train.py

# Keep the container running by starting a shell
exec "$@"
