#! /bin/bash

# exit shell if any command exits with a non-zero exit code
set -e

echo "Entrypoint Started!"

# run the command passed
exec "$@"
