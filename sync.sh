#!/bin/bash

echo "🔄 Starting sync process..."

# Pull latest changes
echo "📥 Pulling latest from GitHub..."
git pull

# Add all changes
echo "📝 Adding new changes..."
git add .

# Commit with timestamp
echo "💾 Committing changes..."
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Auto-sync update: $timestamp"

# Push changes
echo "📤 Pushing to GitHub..."
git push

echo "✅ Sync complete!" 