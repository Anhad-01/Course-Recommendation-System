import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import re

def create_and_save_faiss_index(df_path):
    """
    Create and save a FAISS index for course recommendations.
    
    Args:
        df_path (str): Path to the CSV file containing course data
    """
    # Read the CSV into a pandas DataFrame
    print("Loading course data...")
    df = pd.read_csv(df_path)
    
    # Clean the data - Price column
    print("Cleaning price data...")
    # Fill null values with '0'
    df['Price'] = df['Price'].fillna('0')
    
    # Remove /- and commas from Price column and convert to numeric
    df['Price'] = df['Price'].astype(str).apply(lambda x: re.sub(r'[,/-]', '', x).strip())
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    
    # Fill null values in other columns
    df['Title'] = df['Title'].fillna('')
    df['Offered_by'] = df['Offered_by'].fillna('')
    df['Domain'] = df['Domain'].fillna('')
    df['Duration'] = df['Duration'].fillna('')
    
    # Create combined_features column
    print("Creating combined features...")
    df['combined_features'] = (
        df['Title'].astype(str) + ' ' +
        df['Offered_by'].astype(str) + ' ' +
        df['Domain'].astype(str) + ' ' +
        df['Duration'].astype(str) + ' ' +
        df['Price'].astype(str)
    )
    
    # Initialize sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for combined features
    print("Creating embeddings...")
    embeddings = model.encode(df['combined_features'].tolist())
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    # Save FAISS index
    print("Saving FAISS index...")
    faiss.write_index(index, 'course_index.faiss')
    
    # Save processed DataFrame
    print("Saving processed data...")
    df.to_csv('processed_courses.csv', index=False)
    
    print(f"Successfully created and saved FAISS index with {index.ntotal} courses")
    print(f"Index dimension: {dimension}")


def load_faiss_index():
    """
    Load the FAISS index and processed course data.
    
    Returns:
        tuple: (faiss_index, dataframe) - The loaded FAISS index and processed course DataFrame
    """
    try:
        # Load FAISS index
        index = faiss.read_index('course_index.faiss')
        
        # Load processed DataFrame
        df = pd.read_csv('processed_courses.csv')
        
        print(f"Successfully loaded FAISS index with {index.ntotal} courses")
        return index, df
    
    except FileNotFoundError as e:
        print(f"Error: Required files not found. Please run create_and_save_faiss_index first.")
        print(f"Missing file: {e.filename}")
        return None, None
    except Exception as e:
        print(f"Error loading index or data: {str(e)}")
        return None, None


def search_faiss_index(query, index, df, top_k=10):
    """
    Search for similar courses using the FAISS index.
    
    Args:
        query (str): User query for course recommendations
        index: Loaded FAISS index
        df (pd.DataFrame): DataFrame containing processed course data
        top_k (int): Number of top recommendations to return
    
    Returns:
        list: List of tuples containing (course_dict, similarity_score)
    """
    if index is None or df is None:
        print("Error: Index or DataFrame is None")
        return []
    
    # Initialize sentence transformer model (same as used for indexing)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embedding for the query
    query_embedding = model.encode([query])
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Perform search on FAISS index
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Convert L2 distances to cosine similarity scores
    # For normalized vectors: cosine_similarity = (2 - L2_distance^2) / 2
    similarities = (2 - distances[0] ** 2) / 2
    
    # Prepare results
    recommendations = []
    for i, (idx, similarity) in enumerate(zip(indices[0], similarities)):
        if idx < len(df):  # Ensure index is valid
            course = df.iloc[idx].to_dict()
            recommendations.append((course, float(similarity)))
    
    return recommendations
