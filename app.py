import streamlit as st
import pandas as pd
import os
from recommender_agent import CourseRecommender, create_and_save_faiss_index

# Set page configuration
st.set_page_config(
    page_title="FutureSkills Prime Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .course-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .similarity-score {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .price-tag {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system():
    """Initialize the course recommendation system"""
    csv_path = "FutureSkills Prime_data.csv"
    
    # Check if the processed files exist, if not create them
    if not os.path.exists('course_index.faiss') or not os.path.exists('processed_courses.csv'):
        if os.path.exists(csv_path):
            with st.spinner("Setting up the recommendation system for the first time... This may take a few moments."):
                create_and_save_faiss_index(csv_path)
            st.success("System initialized successfully!")
        else:
            st.error(f"Error: {csv_path} not found in current directory.")
            return None
    
    # Initialize the recommender
    try:
        recommender = CourseRecommender()
        if recommender.is_ready():
            return recommender
        else:
            st.error("Failed to initialize the recommender system.")
            return None
    except Exception as e:
        st.error(f"Error initializing recommender: {str(e)}")
        return None

def display_course_card(course, score, index):
    """Display a course recommendation in a card format"""
    with st.container():
        st.markdown(f"""
        <div class="course-card">
            <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">
                {index}. {course['Title']}
            </h4>
            <p style="margin-bottom: 0.3rem; color: black;">
                <strong style="color: black;">Offered by:</strong> {course['Offered_by']}
            </p>
            <p style="margin-bottom: 0.3rem; color: black;">
                <strong style="color: black;">Domain:</strong> {course['Domain']}
            </p>
            <p style="margin-bottom: 0.3rem; color: black;">
                <strong style="color: black;">Duration:</strong> {course['Duration']}
            </p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span class="price-tag">₹{course['Price']:,}</span>
                <span class="similarity-score">Similarity: {score:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # App header
    st.markdown('<h1 class="main-header">🎓 FutureSkills Prime Course Recommender</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Discover the perfect courses tailored to your learning goals using AI-powered recommendations
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for additional information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This course recommendation system uses:
        - **FAISS** for efficient similarity search
        - **Sentence Transformers** for semantic understanding
        - **Cosine Similarity** for matching courses to your query
        
        Simply enter what you're looking for and get personalized course recommendations!
        """)
        
        st.header("💡 Example Queries")
        st.markdown("""
        - "artificial intelligence machine learning"
        - "web development fullstack programming"
        - "data science analytics"
        - "communication skills leadership"
        - "python programming under 10000"
        - "business courses cheap"
        """)
    
    # Initialize the system
    recommender = initialize_system()
    
    if recommender is None:
        st.stop()
    
    # Main interface
    st.header("🔍 Find Your Perfect Course")
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input for user query
        user_query = st.text_input(
            "Enter your course preferences:",
            placeholder="e.g., Suggest some courses on business under 5000",
            help="Describe what kind of courses you're looking for. You can include topics, price ranges, or specific requirements."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        get_recommendations = st.button(
            "🚀 Get Recommendations",
            type="primary",
            use_container_width=True
        )
    
    # Number of recommendations slider
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=20,
        value=10,
        help="Choose how many course recommendations you'd like to see"
    )
    
    # Process recommendations
    if get_recommendations and user_query.strip():
        with st.spinner("🔍 Finding the best courses for you..."):
            try:
                recommendations = recommender.get_recommendations(user_query, top_k=num_recommendations)
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} course recommendations!")
                    
                    # Display statistics
                    avg_price = sum(course['Price'] for course, _ in recommendations) / len(recommendations)
                    max_similarity = max(score for _, score in recommendations)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Courses Found", len(recommendations))
                    with col2:
                        st.metric("Average Price", f"₹{avg_price:,.0f}")
                    with col3:
                        st.metric("Best Match Score", f"{max_similarity:.4f}")
                    
                    st.markdown("---")
                    st.header("📚 Recommended Courses")
                    
                    # Display recommendations
                    for i, (course, score) in enumerate(recommendations, 1):
                        display_course_card(course, score, i)
                    
                    # Option to download results
                    if st.button("📥 Download Results as CSV"):
                        results_df = pd.DataFrame([
                            {
                                'Rank': i,
                                'Title': course['Title'],
                                'Offered_by': course['Offered_by'],
                                'Domain': course['Domain'],
                                'Duration': course['Duration'],
                                'Price': course['Price'],
                                'Similarity_Score': score
                            }
                            for i, (course, score) in enumerate(recommendations, 1)
                        ])
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"course_recommendations_{user_query.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("No courses found matching your criteria. Try a different search term.")
                    
            except Exception as e:
                st.error(f"An error occurred while getting recommendations: {str(e)}")
    
    elif get_recommendations and not user_query.strip():
        st.warning("Please enter a search query to get recommendations.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        FutureSkills Prime Course Recommender | Powered by AI and Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
