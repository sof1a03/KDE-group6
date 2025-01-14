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

## Here are the key API endpoints:

    Server Management

        GET /$/server - Get server status.

        GET /$/datasets - List all datasets.

        POST /$/datasets - Create a new dataset.

        DELETE /$/datasets/{dataset} - Delete a dataset.

    Dataset Operations

        GET /{dataset}/data - Retrieve data from a dataset.

        POST /{dataset}/data - Add data to a dataset.

        PUT /{dataset}/data - Replace data in a dataset.

        DELETE /{dataset}/data - Delete data from a dataset.

    SPARQL Queries

        GET /{dataset}/query - Execute a SPARQL query.

        POST /{dataset}/query - Execute a SPARQL query (using POST).

        POST /{dataset}/update - Execute a SPARQL update.
