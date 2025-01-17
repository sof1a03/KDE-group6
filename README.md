# KDE-group6

## Installation instructions
1. Unzip the file `dataset/data.zip`
2. Generate `bookData.ttl`; the data knowledge graph, using `dataset/project_trial1.ipynb` and `dataset/prjoect_trial2.ipynb`



## Running the web app
1. `cd web/front-end/` 
2. `npm -i`
3. `ng serve`


## Running The DB
1. `Open apache-jena-fuseki-5.2.0 folder in your command prompt and run the following command`
   `java -jar fuseki-server.jar`
2. `create the DB by basically uploading the owlshelves.ttl`

## Running the API
1. `pip install uvicorn`
2. Choose the root directory
3. `uvicorn api:app --reload`