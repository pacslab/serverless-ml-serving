#!/bin/bash

# load environment variables
export $(cat .env.dev | grep ^[A-Z] | xargs)

# Home directory
export HOME_DIR=${HOME_DIR:-$(pwd)}

docker-compose -f docker-compose-dev.yml up --build

docker-compose -f docker-compose-dev.yml down

