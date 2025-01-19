# KDE-group6

## Installation instructions

1. Download dataset and models from https://drive.google.com/file/d/1PUw8MFHrYdzTgXDpGX5MA9p9RkBe87gO/view?usp=sharing
   Put the contents of the `fuseki` folder into the repository's `/fuseki/` folder
   Put the `data` folder into the root: `/data`
2. Build and run the container:
   `docker-compose up --build`
3. From the host machine, instruct the fuseki container to import the dataset: 
   `docker exec -it fuseki  /bin/bash -c 'cp -r /staging/owlshelvesbig /fuseki/databases/'`
   `docker exec -it fuseki  /bin/bash -c 'cp -r /staging/configs/owlshelvesbig_config.ttl /fuseki/configuration/'`
4. After the import has completed, go back to the running Docker container. Ctrl+C to close, and use the command below to start them back up:
   `docker-compose up`
5. Go to `localhost:4234` to use the app