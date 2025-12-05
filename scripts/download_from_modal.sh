#!/bin/bash
# Download files from Modal volume 'wanda-cache' to local directory

VOLUME_NAME="wanda-cache"
LOCAL_OUT_DIR="./out"
LOCAL_MODEL_DIR="./pruned_models"

echo "Downloading files from Modal volume '$VOLUME_NAME'..."

# Download output files
echo "Downloading output files to $LOCAL_OUT_DIR..."
modal volume get "$VOLUME_NAME" out/ "$LOCAL_OUT_DIR"

# Download pruned models
echo "Downloading pruned models to $LOCAL_MODEL_DIR..."
modal volume get "$VOLUME_NAME" pruned_models/ "$LOCAL_MODEL_DIR"

echo "Download complete!"

