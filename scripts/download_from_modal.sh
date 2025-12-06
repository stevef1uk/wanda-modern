#!/bin/bash
# Download files from Modal volume 'wanda-cache' to local directory

VOLUME_NAME="wanda-cache"
LOCAL_OUT_DIR="./out"
LOCAL_MODEL_DIR="./pruned_models"

echo "Downloading files from Modal volume '$VOLUME_NAME'..."

# Download output files
echo "Downloading output files to $LOCAL_OUT_DIR..."
# Use --force to overwrite existing files
modal volume get "$VOLUME_NAME" out/ "$LOCAL_OUT_DIR" --force

# Download pruned models
echo "Downloading pruned models to $LOCAL_MODEL_DIR..."
# Remove directory if it exists (modal volume get needs to create it)
if [ -d "$LOCAL_MODEL_DIR" ]; then
    echo "Removing existing $LOCAL_MODEL_DIR directory..."
    rm -rf "$LOCAL_MODEL_DIR"
fi
# Download to current directory - modal will create pruned_models/ subdirectory
# The remote path is pruned_models/, so it will create ./pruned_models/ locally
modal volume get "$VOLUME_NAME" pruned_models/ .

echo "Download complete!"

