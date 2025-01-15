from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from pykeen.triples import TriplesFactory
from node2vec import Node2Vec
import torch
import random
from fastapi.middleware.cors import CORSMiddleware


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


# Load TransE model
def load_transe_model():
    """
    Load the pre-trained TransE model from the specified directory.
    """
    global transe_model
    model_path = f"{TRANSE_MODEL_DIR}/trained_model.pkl"# Path to the serialized TransE model
    transe_model = torch.load(model_path, map_location=torch.device("cpu")) # Load the model to CPU
    
    
# Load Node2Vec embeddings
def load_node2vec_embeddings():
    """
    Load precomputed Node2Vec embeddings as a Word2Vec model.
    """
    global node2vec_model
    # Assume embeddings are precomputed and available as a Word2Vec model
    from gensim.models import KeyedVectors # Import KeyedVectors for Word2Vec model
    node2vec_model = KeyedVectors.load_word2vec_format(NODE2VEC_EMBEDDINGS_PATH, binary=False) # Load embeddings
    
    
# Helper to execute SPARQL queries
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
    if not transe_model:  # Ensure the TransE model is loaded
        raise HTTPException(status_code=500, detail="TransE model not loaded")

    entity_to_id = transe_model.entity_to_id # Retrieve entity-to-ID mapping from the model
    if user_id not in entity_to_id:  # Check if the user exists in the model
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get the embedding for the user
    user_embedding = transe_model.entity_representations[0](
        torch.tensor([entity_to_id[user_id]])
    ).detach().numpy().squeeze()

    ''' ATTENTION: I'M NOT SURE THIS PART IS CORRECT, NEED TO CHECK WITH THE TEAM'''
    similarities = [] # List to store similarity scores
    for entity, idx in entity_to_id.items():
        if entity.startswith("http://example.org/book"):  # Filter only books
            entity_embedding = transe_model.entity_representations[0](
                torch.tensor([idx])
            ).detach().numpy().squeeze()
            similarity = -((user_embedding - entity_embedding) ** 2).sum()
            similarities.append((entity, similarity))

    # Sort by similarity and return the top N books
    top_books = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return [book for book, _ in top_books]


# Node2Vec-based similar book recommendations
def predict_similar_books_node2vec(book_id, top_n=5):
    """
    Recommend books similar to a given book using Node2Vec embeddings.
    Args:
        book_id (str): The ID of the book.
        top_n (int): Number of similar books to return.
    Returns:
        list: A list of similar book IDs.
    """
    if not node2vec_model: # Ensure the Node2Vec model is loaded
        raise HTTPException(status_code=500, detail="Node2Vec model not loaded")

    try:
        similar_books = node2vec_model.most_similar(book_id, topn=top_n)
        return [book for book, _ in similar_books] # Return only the book IDs
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found in Node2Vec embeddings")

# API endpoint: Fetch personalized recommendations
@app.get("/api/recommended_books")
def recommended_books(userid: str, top_n: int = 5):
    """
    Fetch personalized book recommendations for a user using TransE and demographics.
    Args:
        userid (str): The ID of the user.
        top_n (int): Number of recommendations to return.
    Returns:
        dict: A dictionary with the user ID and recommended books.
    """
    try:
        # Get recommendations using TransE
        recommendations = predict_top_books_transe(userid, top_n=top_n)
        return {"userid": userid, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# API endpoint: Fetch books similar to a given book
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
    # 1. Construct the initial query to fetch distinct book URIs
    query = """
    PREFIX ex: <http://example.org/owlshelves#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?book WHERE {
        ?book ex:hasTitle ?title .
    """

    # ... (Add filters for isbn, title, author, publisher, categories) ...

    if start_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year >= {start_year}) .'

    if end_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year <= {end_year}) .'

    query += f"}} LIMIT {pageSize} OFFSET {(pageNum - 1) * pageSize}"
    print(query)

    try:
        results = execute_sparql_query(query)
        book_uris = [result['book']['value'] for result in results["results"]["bindings"]]

        books = []
        for book_uri in book_uris:
            # 2. Construct a separate query for each book URI to fetch details
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

            # 3. Extract the book details from the results
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
    
# API endpoint: Fetch random book recommendations
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

        if userid: # If a user ID is provided, exclude liked books
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

        random.shuffle(available_books) # Shuffle books for randomness
        return {"surprise_me": available_books[:top_n]} # Return random recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/api/categories")  # New categories endpoint
def get_categories():
    """
    Returns the list of categories.
    """
    return {"categories": categories[:1000]}


# Load models on startup
@app.on_event("startup")
def load_models():
    """
    Load models (TransE and Node2Vec) when the application starts.
    """
    load_transe_model()  # Load the TransE model
    load_node2vec_embeddings()  # Load the Node2Vec embeddings

@app.on_event("startup")
def retrieve_categories():  # Function name updated
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
