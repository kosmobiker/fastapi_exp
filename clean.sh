#!/bin/bash

sudo systemctl restart docker.socket docker.service
# Stop all containers
docker stop $(docker ps -aq)

# Delete all containers
docker rm $(docker ps -aq)

# Delete all images
docker rmi $(docker images -q)

# Delete all volumes
docker volume rm $(docker volume ls -q)

#Delete all networks
docker network rm $(docker network ls -q)