#!/bin/bash

docker restart $1
docker exec $1 streamlit run ./app/main.py --server.port=8080 --server.address=0.0.0.0
