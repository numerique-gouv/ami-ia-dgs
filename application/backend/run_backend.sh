#!/usr/bin/env bash

#########################
#  Run docker
#
#  Options:
#     --bash : runs the container with bash as entrypoint
#     --port : configure port
#     --no-front : do not activate front
#########################


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
    "--bash") set -- "$@" "-b" ;;
    "--port") set -- "$@" "-p" ;;
    "--no-front") set -- "$@" "-n";;
    *)        set -- "$@" "$arg"
  esac
done

## get call arguments
RUN_BASH=0
PORT=5000
WITH_FRONT="True"
while getopts :bp:n option
do
	case "${option}"
	in
	b) RUN_BASH=1;;
	p) PORT=${OPTARG};;
  n) WITH_FRONT="False";;
	\?) echo "Option not supported: ${OPTARG}";;
	esac
done

CONTAINER=starclay/dgs_backend:latest

APP_LOG_LEVEL=DEBUG
echo ${WITH_FRONT}


if [[ "${RUN_BASH}" -eq "1" ]]; then
    docker run -it \
            -p $PORT:5000 \
            -e APP_LOG_LEVEL=${APP_LOG_LEVEL} \
            -e APP_ACTIVATE_FRONT=${WITH_FRONT} \
            -v $(pwd)/src/data:/dgs_backend/data \
            -v $(pwd)/src/password.json:/dgs_backend/password.json \
            ${CONTAINER} bash
else
    docker run \
            -p $PORT:5000 \
            --network="bridge" \
            -e APP_LOG_LEVEL=${APP_LOG_LEVEL} \
            -e APP_ACTIVATE_FRONT=${WITH_FRONT} \
            -v $(pwd)/src/data:/dgs_backend/data \
            -v $(pwd)/src/password.json:/dgs_backend/password.json \
            ${CONTAINER}
fi



#docker run -it --cpus=4 --gpus all -p 8888:8888 -p 27017:27017 octave_intent_matcher:latest bash