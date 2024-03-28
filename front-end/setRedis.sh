#!/bin/bash

# Check for the directory and file arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <directory> <file>"
    exit 1
fi

DIRECTORY=$1
FILE=$2

# Name of the JSON file
# FILE="input.json"

# Full path to the JSON file
FILE_PATH="${DIRECTORY%/}/$FILE"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "File not found: $FILE_PATH"
    exit 1
fi

# Fetch current IP address
CURRENT_IP=$(curl -s https://api.ipify.org)

# Check if the IP address was fetched successfully
if [[ -z "$CURRENT_IP" ]]; then
    echo "Failed to fetch the current IP address."
    exit 1
fi

# Update the IP address in the JSON file
jq --arg ip "redis://$CURRENT_IP" '.["$composer"].redis.uri = $ip' "$FILE_PATH" > "${DIRECTORY%/}/temp.json" && mv "${DIRECTORY%/}/temp.json" "$FILE_PATH"

# Check if jq successfully updated the file
if [[ $? -ne 0 ]]; then
    echo "Failed to update the JSON file."
    exit 1
fi

echo "The IP address has been updated successfully in $FILE_PATH."
