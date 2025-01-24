# KDE-group6

## Installation instructions

1. Make sure Docker is installed on your system and running. Refer to https://www.docker.com/
2. Download dataset and models from https://drive.google.com/file/d/1PUw8MFHrYdzTgXDpGX5MA9p9RkBe87gO/view?usp=sharing
   Put the contents of the unzipped folders' `fuseki/staging` location, into the repository's `fuseki/staging` location
   Put the `data` folder into the root: `/data`
3. Build and run the container:
   `docker-compose up --build`
4. From the host machine, instruct the fuseki container to import the dataset: 
   `docker exec -it fuseki  /bin/bash -c 'cp -r /staging/owlshelvesbig /fuseki/databases/'`
   `docker exec -it fuseki  /bin/bash -c 'cp -r /staging/configs/owlshelvesbig_config.ttl /fuseki/configuration/'`
5. After the import has completed, go back to the running Docker container. Ctrl+C to close, and use the command below to start them back up:
   `docker-compose up`
6. Go to `localhost:4234` to use the app

# YOUTUBE INSTRUCTIONS
https://www.youtube.com/watch?v=yAOon0DWW7A

## Data instructions
Because within Apache Jena's Docker container the dataset upload mechanism is broken, the main instructions involve you copying the raw data files over into the container's appropriate folder.
This section contains instructions on how to generate the dataset, as well as how to generate the files seen in step 2 of the installation instructions.
1. Cleaning of the Data: From the original datasets please add in your work enviroment the merged_invalid_books csv file. Afterwards it will be possible to clean the dataset and to retrieve a functional one
2. <Open Library linking part>
3. Given the linked Datasets (that are also available, already linked, in the 'linked data' file, it is possible to run the 'transformation.py' code in order to obtain the ttl file for the next steps and a statystical analysis of the obtained file.
4. Download the regular, non-containerized version of Apache Jena Fuseki from here: https://jena.apache.org/download/
5. Run the jar using `java -jar fuseki-server.jar`. This part requires you have java installed, and that port 3030 must be free on your machine.
6. Navigate to `localhost:3030` within a web-browser and click on 'Add one'. Give the database the name of 'owlshelvesbig' and select the TDB2 option.  Upload the aforementioned newly generated .ttl file. Click on 'Add Data', 'Select files', and upload the aforementioned newly generated .ttl file. Click 'Upload now' and wait for all triples to upload
7. Relative to the Apache Jena folder, navigate to `'./run/databases/owlshelvesbig`. Replace the folder with the same name (located in `fuseki/staging` of the downloaded folder)
8. Relative to the Apache Jena folder, navigate to `'./run/configuration` and open the configuration file. Alter the path in the line that starts with `tdb2:location` to have `"/fuseki/databases/owlshelvesbig"` as its value. Rename this file to `owlshelvesbig_config.ttl` and copy it into `fuseki/staging/configs` and replace the old file.
9. You can now continue from step 3 of the main instructions.
