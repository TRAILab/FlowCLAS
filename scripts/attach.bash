#!/bin/bash

CONTAINER_NAME=$1
sudo bash -c "echo '' > $(docker inspect --format="{{.LogPath}}" $CONTAINER_NAME)"
docker attach --detach-keys="ctrl-a" $CONTAINER_NAME