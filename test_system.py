#!/usr/bin/env python3
"""
Test script for the FutureSkills Course Recommender system.
This script tests the core functionality and provides example usage.
"""

import os
import sys
from recommender_agent import CourseRecommender, create_and_save_faiss_index

def test_system():
    """Test the course recommendation system"""
    
    print("="*60)
    print("FUTURESKILLS COURSE RECOMMENDER - SYSTEM TEST")
    print("="*60)
    
    # Check if CSV file exists
    csv_path = "FutureSkills Prime_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print("Please ensure the CSV file is in the current directory.")
        return False
    
    print(f"✅ Found dataset: {csv_path}")
    
    # Create FAISS index if it doesn't exist
    if not os.path.exists('course_index.faiss') or not os.path.exists('processed_courses.csv'):
        print("🔄 Creating FAISS index for the first time...")
        try:
            create_and_save_faiss_index(csv_path)
            print("✅ FAISS index created successfully!")
        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False
    else:
        print("✅ FAISS index already exists")
    
    # Initialize recommender
    print("🔄 Initializing recommender...")
    try:
        recommender = CourseRecommender()
        if not recommender.is_ready():
            print("❌ Failed to initialize recommender")
            return False
        print("✅ Recommender initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing recommender: {e}")
        return False
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "web development",
        "data science",
        "python programming",
        "business communication"
    ]
    
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION FUNCTIONALITY")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        try:
            recommendations = recommender.get_recommendations(query, top_k=3)
            if recommendations:
                print(f"✅ Found {len(recommendations)} recommendations")
                for j, (course, score) in enumerate(recommendations, 1):
                    print(f"   {j}. {course['Title']} (Score: {score:.4f})")
            else:
                print("⚠️  No recommendations found")
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! System is working correctly.")
    print("="*60)
    print("\nTo run the web interface:")
    print("  streamlit run app.py")
    print("\nTo run individual tests:")
    print("  python recommender_agent.py")
    
    return True

if __name__ == "__main__":
    success = test_system()
    if not success:
        sys.exit(1)
