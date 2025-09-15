#!/bin/bash

# Script to monitor backend logs with filtering options

echo "=== Computer Vision Backend Log Monitor ==="
echo "Use Ctrl+C to stop monitoring"
echo ""

# Default filter is empty (show all logs)
FILTER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --video)
      FILTER="VIDEO"
      echo "Filtering for video processing logs..."
      ;;
    --processing)
      FILTER="PROCESSING"
      echo "Filtering for processing logs..."
      ;;
    --api)
      FILTER="API"
      echo "Filtering for API request logs..."
      ;;
    --errors)
      FILTER="ERROR"
      echo "Filtering for error logs..."
      ;;
    --help)
      echo "Usage: monitor-logs.sh [--video|--processing|--api|--errors]"
      echo ""
      echo "Options:"
      echo "  --video       Show video processing logs only"
      echo "  --processing  Show processing step logs only"
      echo "  --api         Show API request logs only"
      echo "  --errors      Show error logs only"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
  shift
done

# Function to colorize log levels
colorize_logs() {
  if [ -n "$FILTER" ]; then
    grep --line-buffered "$FILTER" | sed -E \
      -e 's/INFO/\x1b[32mINFO\x1b[0m/g' \
      -e 's/WARNING/\x1b[33mWARNING\x1b[0m/g' \
      -e 's/ERROR/\x1b[31mERROR\x1b[0m/g' \
      -e 's/\[VIDEO\]/\x1b[35m[VIDEO]\x1b[0m/g' \
      -e 's/\[PROCESSING\]/\x1b[36m[PROCESSING]\x1b[0m/g' \
      -e 's/\[API\]/\x1b[34m[API]\x1b[0m/g'
  else
    sed -E \
      -e 's/INFO/\x1b[32mINFO\x1b[0m/g' \
      -e 's/WARNING/\x1b[33mWARNING\x1b[0m/g' \
      -e 's/ERROR/\x1b[31mERROR\x1b[0m/g' \
      -e 's/\[VIDEO\]/\x1b[35m[VIDEO]\x1b[0m/g' \
      -e 's/\[PROCESSING\]/\x1b[36m[PROCESSING]\x1b[0m/g' \
      -e 's/\[API\]/\x1b[34m[API]\x1b[0m/g'
  fi
}

echo "Starting log monitoring..."
echo "==========================================="

# Monitor logs with color coding
docker compose logs -f backend 2>&1 | colorize_logs
