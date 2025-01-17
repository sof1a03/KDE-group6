from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from pykeen.triples import TriplesFactory
from node2vec import Node2Vec
import torch
import rdflib
import random
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

global triples_factory

app = FastAPI()

origins = [
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration for GraphDB and file paths
GRAPHDB_SPARQL_ENDPOINT = "http://localhost:3030/owlshelvesbig/sparql" #needs to put the db endpoint
TRANSE_MODEL_DIR = "transe_model_output" # Directory where the TransE model is stored
NODE2VEC_EMBEDDINGS_PATH = "node2vec_embeddings.vec" # File path to Node2Vec embeddings
DEFAULT_PAGE_SIZE = 10  # Default number of results per page for pagination


# Initialize TransE and Node2Vec models
transe_model = None
node2vec_model = None
categories = []


def load_rdf_as_triples(rdf_file_path):
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(rdf_file_path, format="turtle") 
    triples = [(str(subj), str(pred), str(obj)) for subj, pred, obj in rdf_graph]
    return triples
rdf_file_path = "./OwlshelvesFinal_RDF.ttl"
triples_factory_path = "./triples_factory.pkl"

"""
try:
    print("Loading pre-computed triples factory...")
    with open(triples_factory_path, "rb") as f:
        triples_factory = pickle.load(f)
    print("Triples factory loaded successfully!")

except FileNotFoundError:
    print("Pre-computed triples factory not found. Loading triples...")
    triples = load_rdf_as_triples(rdf_file_path)
    triples_array = np.array([[s, p, o] for s, p, o in triples], dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(triples=triples_array)

    print("Saving triples factory for future use...")
    with open(triples_factory_path, "wb") as f:
        pickle.dump(triples_factory, f)
    print("Triples factory saved successfully!")
    
"""

# Load TransE model
def load_transe_model():
    """
    Load the pre-trained TransE model from the specified directory.
    """
    global transe_model
    model_path = f"{TRANSE_MODEL_DIR}/trained_model.pkl"# Path to the serialized TransE model
    transe_model = torch.load(model_path, map_location=torch.device("cpu")) # Load the model to CPU
    
    
def load_node2vec_embeddings():
    """
    Load precomputed Node2Vec embeddings as a Word2Vec model.
    """
    global node2vec_model
    from gensim.models import KeyedVectors # Import KeyedVectors for Word2Vec model
    node2vec_model = KeyedVectors.load_word2vec_format(NODE2VEC_EMBEDDINGS_PATH, binary=False) # Load embeddings
    
    
def execute_sparql_query(query: str):
    """
    Execute a SPARQL query against the GraphDB endpoint.
    Args:
        query (str): The SPARQL query string.
    Returns:
        dict: Parsed JSON response from GraphDB.
    """
    sparql = SPARQLWrapper(GRAPHDB_SPARQL_ENDPOINT)# Initialize SPARQL wrapper with the endpoint URL
    sparql.setQuery(query) # Set the SPARQL query
    sparql.setReturnFormat(JSON)# Execute the query and parse the JSON response
    try:
        response = sparql.query().convert() # Execute the query and parse the JSON response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL query failed: {e}")  # Handle errors


# TransE-based recommendation logic
def predict_top_books_transe(user_id, top_n=5):
    """
    Predict the top book recommendations for a user based on TransE embeddings.
    Args:
        user_id (str): The ID of the user.
        top_n (int): Number of recommendations to return.
    Returns:
        list: A list of recommended book IDs.
    """
    if not transe_model:  
        raise HTTPException(status_code=500, detail="TransE model not loaded")

    entity_to_id = transe_model.entity_to_id 
    if user_id not in entity_to_id:  
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    user_embedding = transe_model.entity_representations[0](
        torch.tensor([entity_to_id[user_id]])
    ).detach().numpy().squeeze()

    ''' ATTENTION: I'M NOT SURE THIS PART IS CORRECT, NEED TO CHECK WITH THE TEAM'''
    similarities = [] 
    for entity, idx in entity_to_id.items():
        if entity.startswith("http://example.org/book"): 
            entity_embedding = transe_model.entity_representations[0](
                torch.tensor([idx])
            ).detach().numpy().squeeze()
            similarity = -((user_embedding - entity_embedding) ** 2).sum()
            similarities.append((entity, similarity))

    top_books = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return [book for book, _ in top_books]

def predict_similar_books(book_id, triples_factory, model, top_n=5):
    """
    Predict books similar to the given book_id based on embeddings.
    """
    entity_embeddings = model.entity_representations[0]
    all_entities = list(triples_factory.entity_to_id.keys())
    
    if book_id not in triples_factory.entity_to_id:
        raise ValueError(f"Book ID {book_id} not found in the entity set!")
    
    book_idx = triples_factory.entity_to_id[book_id]
    book_embedding = entity_embeddings(torch.tensor(book_idx)).detach().numpy()
    
    similarities = []
    for entity in all_entities:
        if "book" in entity:  # Ensure only books are considered
            entity_idx = triples_factory.entity_to_id[entity]
            entity_embedding = entity_embeddings(torch.tensor(entity_idx)).detach().numpy()
            similarity = -((book_embedding - entity_embedding) ** 2).sum()  # Euclidean distance
            similarities.append((entity, similarity))
    
    # Sort by similarity and return top_n similar books
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [entity for entity, _ in similarities[:top_n]]

def predict_similar_books_node2vec(book_id, top_n=5):
    """
    Recommend books similar to a given book using Node2Vec embeddings.
    Args:
        book_id (str): The ID of the book.
        top_n (int): Number of similar books to return.
    Returns:
        list: A list of similar book IDs.
    """
    if not node2vec_model: 
        raise HTTPException(status_code=500, detail="Node2Vec model not loaded")

    try:
        similar_books = node2vec_model.most_similar(book_id, topn=top_n)
        return [book for book, _ in similar_books] # Return only the book IDs
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found in Node2Vec embeddings")

def compute_virtual_user_embedding(book_ids, triples_factory, model):
    entity_embeddings = model.entity_representations[0]
    
    book_indices = [
        triples_factory.entity_to_id[book_id] for book_id in book_ids if book_id in triples_factory.entity_to_id
    ]
    if not book_indices:
        raise ValueError("None of the provided book IDs exist in the entity set!")
    
    # Get embeddings for the provided books
    book_embeddings = torch.stack([entity_embeddings(torch.tensor(idx)) for idx in book_indices])
    
    # Compute virtual user embedding (e.g., mean of book embeddings)
    virtual_user_embedding = book_embeddings.mean(dim=0)
    return virtual_user_embedding

def predict_top_books_for_virtual_user(virtual_user_embedding, triples_factory, model, top_n=5):
    entity_embeddings = model.entity_representations[0]
    all_entities = list(triples_factory.entity_to_id.keys())
    
    similarities = []
    for entity in all_entities:
        if "book" in entity:  # Ensure only books are considered
            entity_idx = triples_factory.entity_to_id[entity]
            entity_embedding = entity_embeddings(torch.tensor(entity_idx)).detach().numpy()
            similarity = -((virtual_user_embedding.detach().numpy() - entity_embedding) ** 2).sum()
            similarities.append((entity, similarity))
    
    # Sort by similarity and return top_n books
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [entity for entity, _ in similarities[:top_n]]

@app.get("/api/recommended_books")
def recommended_books(book_ids: List[str] = Query(None), top_n: int = 5):
    """
    Fetch personalized book recommendations for a user using TransE and demographics.
    Args:
        book_ids (str): IDs of books the user likes.
        top_n (int): Number of recommendations to return.
    Returns:
        list: Book ids of recommended books.
    """
    global triples_factory
    for i in range(len(book_ids)):
        book_ids[i] = f"http://example.org/book_{book_ids[i]}"
    virtual_user_embedding = compute_virtual_user_embedding(book_ids, triples_factory, transe_model)
    try:
        recommended_books = predict_top_books_for_virtual_user(virtual_user_embedding, triples_factory, transe_model, top_n=top_n)
        return recommended_books
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/similar_books")
def similar_books(bookid: str, top_n: int = 5):
    """
    Fetch books similar to the given book using Node2Vec embeddings.
    Args:
        bookid (str): The ID of the book.
        top_n (int): Number of similar books to return.
    Returns:
        dict: A dictionary with the book ID and similar books.
    """
    try:
        recommendations = predict_similar_books_node2vec(bookid, top_n=top_n) # Get similar books
        return {"bookid": bookid, "similar_books": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_books(
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    publisher: Optional[str] = None,
    categories: Optional[List[str]] = Query(None),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pageSize: int = 10,
    pageNum: int = 1,
):
    query = """
    PREFIX ex: <http://example.org/owlshelves#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?book WHERE {
        ?book ex:hasTitle ?title .
        ?book ex:hasBookCover ?cover .
    """

    if isbn:
        query += f'?book ex:hasISBN "{isbn}"^^xsd:string .'

    if title:
        query += f'FILTER regex(?title, "{title}", "i") .'

    if author:
        query += f'FILTER regex(?author, "{author}", "i") .'

    if publisher:
        query += f'FILTER regex(?publisher, "{publisher}", "i") .'

    if categories:
        for category in categories:
            query += f'?book ex:hasGenre "{category}"^^xsd:string .'  


    if start_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year >= {start_year}) .'

    if end_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year <= {end_year}) .'

    query += "}"    
    query += f" LIMIT {pageSize} OFFSET {(pageNum - 1) * pageSize}"

    try:
        print(query)
        results = execute_sparql_query(query)
        book_uris = [result['book']['value'] for result in results["results"]["bindings"]]

        books = []
        for book_uri in book_uris:
            book_query = f"""
            PREFIX ex: <http://example.org/owlshelves#>
            SELECT ?title ?author ?ISBN ?publisher ?year ?genre ?cover WHERE {{
                <{book_uri}> ex:hasTitle ?title .
                OPTIONAL {{ <{book_uri}> ex:hasAuthor ?author . }}
                OPTIONAL {{ <{book_uri}> ex:hasISBN ?ISBN . }}
                OPTIONAL {{ <{book_uri}> ex:hasPublisher ?publisher . }}
                OPTIONAL {{ <{book_uri}> ex:hasYearOfPublication ?year . }}
                OPTIONAL {{ <{book_uri}> ex:hasGenre ?genre . }}
                OPTIONAL {{ <{book_uri}> ex:hasBookCover ?cover . }}
            }}
            """

            book_results = execute_sparql_query(book_query)

            book_data = book_results["results"]["bindings"]
            book = {
                "bookid": book_uri,
                "name": book_data[0]["title"]["value"] if book_data else None,
                "author": book_data[0]["author"]["value"] if book_data and "author" in book_data[0] else None,
                "ISBN": book_data[0]["ISBN"]["value"] if book_data and "ISBN" in book_data[0] else None,
                "publisher": book_data[0]["publisher"]["value"] if book_data and "publisher" in book_data[0] else None,
                "year": book_data[0]["year"]["value"] if book_data and "year" in book_data[0] else None,
                "url": book_data[0]["cover"]["value"] if book_data and "cover" in book_data[0] else None,
                "genres": [result["genre"]["value"] for result in book_data if "genre" in result],
            }
            books.append(book)

        return books

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/surprise_me")
def surprise_me(userid: Optional[str] = None, top_n: int = 5):
    """
    Fetch random book recommendations, excluding books already liked by the user.
    Args:
        userid (str, optional): The ID of the user.
        top_n (int): Number of random books to return.
    Returns:
        dict: A dictionary of random book recommendations.
    """
    try:
        query = "SELECT ?book WHERE { ?book a <http://example.org/Book> }"
        all_books = execute_sparql_query(query)["results"]["bindings"]

        if userid: 
            liked_query = f"""
            SELECT ?book WHERE {{
                <http://example.org/user/{userid}> <http://example.org/likesBook> ?book .
            }}
            """
            liked_books = execute_sparql_query(liked_query)["results"]["bindings"]
            liked_book_ids = {b["book"]["value"] for b in liked_books}
            available_books = [b["book"]["value"] for b in all_books if b["book"]["value"] not in liked_book_ids]
        else:
            available_books = [b["book"]["value"] for b in all_books]

        random.shuffle(available_books) 
        return {"surprise_me": available_books[:top_n]} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/api/categories")
def get_categories():
    """
    Returns the list of categories.
    """
    return categories[:1000]


@app.on_event("startup")
def load_models():
    """
    Load models (TransE and Node2Vec) when the application starts.
    """
    load_transe_model()  
    load_node2vec_embeddings() 

@app.on_event("startup")
def retrieve_categories():  
  """Executes the SPARQL query and stores the results in the global variable."""
  global categories

  query = """
    PREFIX ex: <http://example.org/owlshelves#>

    SELECT ?genre (COUNT(?book) AS ?bookCount) WHERE {
      ?book ex:hasGenre ?genre .
    }
    GROUP BY ?genre 
    HAVING (?bookCount > 1) 
    ORDER BY desc(?bookCount)
  """
  results = execute_sparql_query(query)

  for result in results["results"]["bindings"]:
    genre = result["genre"]["value"]
    categories.append(genre)
