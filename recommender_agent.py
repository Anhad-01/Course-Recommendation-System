import os
from cosine_faiss_utils import create_and_save_faiss_index, load_faiss_index, search_faiss_index

class CourseRecommender:
    """
    A course recommendation agent that uses FAISS for similarity search.
    """
    
    def __init__(self):
        """
        Initialize the CourseRecommender by loading the FAISS index and processed course data.
        """
        print("Initializing Course Recommender...")
        self.index, self.df = load_faiss_index()
        
        if self.index is None or self.df is None:
            print("Warning: Failed to load FAISS index or course data.")
            print("Please ensure that create_and_save_faiss_index() has been run first.")
        else:
            print(f"Course Recommender initialized successfully with {len(self.df)} courses.")
    
    def get_recommendations(self, query, top_k=10):
        """
        Get course recommendations based on a user query.
        
        Args:
            query (str): User query describing desired course characteristics
            top_k (int): Number of top recommendations to return (default: 10)
        
        Returns:
            list: List of tuples containing (course_dict, similarity_score)
        """
        if self.index is None or self.df is None:
            print("Error: Recommender not properly initialized. Cannot provide recommendations.")
            return []
        
        print(f"Searching for courses similar to: '{query}'")
        recommendations = search_faiss_index(query, self.index, self.df, top_k)
        
        if recommendations:
            print(f"Found {len(recommendations)} recommendations:")
            for i, (course, score) in enumerate(recommendations, 1):
                print(f"{i}. {course['Title']} by {course['Offered_by']} "
                      f"(Similarity: {score:.4f})")
        else:
            print("No recommendations found.")
        
        return recommendations
    
    def display_recommendation_details(self, recommendations):
        """
        Display detailed information about the recommended courses.
        
        Args:
            recommendations (list): List of tuples containing (course_dict, similarity_score)
        """
        if not recommendations:
            print("No recommendations to display.")
            return
        
        print("\n" + "="*80)
        print("DETAILED COURSE RECOMMENDATIONS")
        print("="*80)
        
        for i, (course, score) in enumerate(recommendations, 1):
            print(f"\n{i}. Title: {course['Title']}")
            print(f"   Offered by: {course['Offered_by']}")
            print(f"   Domain: {course['Domain']}")
            print(f"   Duration: {course['Duration']}")
            print(f"   Price: ₹{course['Price']}")
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Combined Features: {course.get('combined_features', 'N/A')[:100]}...")
            print("-" * 80)
    
    def is_ready(self):
        """
        Check if the recommender is ready to provide recommendations.
        
        Returns:
            bool: True if the recommender is properly initialized, False otherwise
        """
        return self.index is not None and self.df is not None


def main():
    """
    Main function to demonstrate the course recommendation system.
    """
    # File path to the original CSV
    csv_path = "FutureSkills Prime_data.csv"
    
    # Check if the processed files exist, if not create them
    if not os.path.exists('course_index.faiss') or not os.path.exists('processed_courses.csv'):
        print("Creating FAISS index from course data...")
        if os.path.exists(csv_path):
            create_and_save_faiss_index(csv_path)
        else:
            print(f"Error: {csv_path} not found in current directory.")
            return
    
    # Initialize the recommender
    recommender = CourseRecommender()
    
    if not recommender.is_ready():
        print("Failed to initialize the recommender. Exiting.")
        return
    
    # Example usage
    print("\n" + "="*60)
    print("COURSE RECOMMENDATION SYSTEM DEMO")
    print("="*60)
    
    # Example queries
    example_queries = [
        "artificial intelligence machine learning",
        "web development fullstack programming",
        "data science analytics",
        "communication skills leadership",
        "python programming"
    ]
    
    for query in example_queries:
        print(f"\n{'='*40}")
        print(f"Query: '{query}'")
        print('='*40)
        
        recommendations = recommender.get_recommendations(query, top_k=5)
        
        if recommendations:
            print(f"\nTop 3 recommendations for '{query}':")
            for i, (course, score) in enumerate(recommendations[:3], 1):
                print(f"{i}. {course['Title']} by {course['Offered_by']} "
                      f"(Score: {score:.4f}, Price: ₹{course['Price']})")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
