#!/bin/sh

# Run the training script
python pyfiles/train.py

# python pyfiles/pipeline.py

# python pyfiles/schedule.py

# Keep the container running by starting a shell
exec "$@"
