#!/bin/bash
set -e

echo "Waiting for web service to complete migrations..."
sleep 5

echo "Starting backend service..."
exec python -m backend.sync_worker
