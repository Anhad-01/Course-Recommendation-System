# FutureSkills Prime Course Recommender

A sophisticated AI-powered course recommendation system that uses semantic search and FAISS indexing to provide personalized course suggestions based on user queries.

## 🌟 Features

- **Semantic Search**: Uses sentence transformers to understand the meaning of user queries
- **Fast Similarity Search**: Leverages FAISS (Facebook AI Similarity Search) for efficient vector operations
- **Interactive Web Interface**: Beautiful Streamlit-based frontend with modern UI
- **Real-time Recommendations**: Get instant course suggestions based on your preferences
- **Detailed Course Information**: View comprehensive details including price, duration, domain, and provider
- **Similarity Scoring**: Each recommendation comes with a confidence score
- **Export Functionality**: Download your recommendations as CSV files

## 🏗️ Project Structure

```
futureskills_course_recommendation/
├── FutureSkills Prime_data.csv     # Original course dataset
├── cosine_faiss_utils.py          # Core utility functions for FAISS operations
├── recommender_agent.py           # Course recommender class and logic
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── course_index.faiss            # Generated FAISS index (created on first run)
└── processed_courses.csv         # Processed course data (created on first run)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   cd futureskills_course_recommendation
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 💻 Usage

### Web Interface (Recommended)

1. **Launch the app**: Run `streamlit run app.py`
2. **Enter your query**: Type what kind of courses you're looking for
   - Example: "artificial intelligence machine learning"
   - Example: "web development courses under 10000"
   - Example: "business communication skills"
3. **Adjust settings**: Use the slider to choose how many recommendations you want
4. **Get recommendations**: Click the "Get Recommendations" button
5. **Review results**: Browse through the recommended courses with similarity scores
6. **Download results**: Optionally download your recommendations as a CSV file

### Command Line Interface

You can also run the recommender directly:

```bash
python recommender_agent.py
```

This will create the FAISS index (if it doesn't exist) and run example queries.

### Programmatic Usage

```python
from recommender_agent import CourseRecommender

# Initialize the recommender
recommender = CourseRecommender()

# Get recommendations
recommendations = recommender.get_recommendations("python programming", top_k=5)

# Display results
for course, score in recommendations:
    print(f"{course['Title']} - Similarity: {score:.4f}")
```

## 🔧 Technical Details

### Core Components

#### 1. **cosine_faiss_utils.py**
- `create_and_save_faiss_index()`: Processes the CSV data and creates FAISS index
- `load_faiss_index()`: Loads the pre-built FAISS index and processed data
- `search_faiss_index()`: Performs similarity search for user queries

#### 2. **recommender_agent.py**
- `CourseRecommender` class: Main recommendation engine
- Handles initialization, query processing, and result formatting

#### 3. **app.py**
- Streamlit web application with modern UI
- Interactive course discovery and recommendation display
- Export functionality and responsive design

### Algorithm Overview

1. **Data Preprocessing**:
   - Cleans price data (removes formatting, handles nulls)
   - Creates combined feature strings from course attributes
   - Generates embeddings using 'all-MiniLM-L6-v2' model

2. **Index Creation**:
   - Uses FAISS IndexFlatL2 for exact similarity search
   - Normalizes embeddings for cosine similarity computation
   - Stores index for fast retrieval

3. **Query Processing**:
   - Converts user queries to embeddings using the same model
   - Performs similarity search in the FAISS index
   - Converts L2 distances to cosine similarity scores
   - Returns ranked recommendations

### Similarity Scoring

The system uses cosine similarity, where:
- **1.0**: Perfect match
- **0.8-0.99**: Very high similarity
- **0.6-0.79**: Good similarity
- **0.4-0.59**: Moderate similarity
- **< 0.4**: Low similarity

## 📊 Dataset

The system works with the `FutureSkills Prime_data.csv` dataset containing:
- **Title**: Course name
- **Offered_by**: Course provider/institution
- **Domain**: Subject area or category
- **Duration**: Course length
- **Price**: Course cost (cleaned and normalized)

## 🎯 Example Queries

Try these example queries to explore the system:

- `"artificial intelligence machine learning"`
- `"web development fullstack programming"`
- `"data science analytics"`
- `"communication skills leadership"`
- `"python programming under 10000"`
- `"business courses cheap"`
- `"certification courses duration 100 hours"`

## ⚙️ Configuration

### Customizing Search Parameters

You can modify the search behavior by adjusting parameters in the code:

- **top_k**: Number of recommendations to return
- **Model**: Change the sentence transformer model in both files (currently 'all-MiniLM-L6-v2')
- **FAISS Index Type**: Modify to use different FAISS index types for different trade-offs

### Performance Optimization

- **IndexIVFFlat**: For larger datasets, consider using IVF (Inverted File) indexes
- **IndexHNSW**: For approximate but faster search
- **GPU Support**: Use `faiss-gpu` for GPU acceleration with large datasets

## 🐛 Troubleshooting

### Common Issues

1. **"Required files not found"**: Run the system once to generate the FAISS index
2. **Memory errors**: Reduce batch size or use approximate search methods
3. **Slow performance**: Ensure you're using the CPU-optimized FAISS version
4. **Import errors**: Make sure all dependencies are installed correctly

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 1GB free space for models and indexes
- **CPU**: Multi-core processor recommended for faster embedding generation

## 🔮 Future Enhancements

- **Advanced Filtering**: Add price range, duration, and domain filters
- **User Profiles**: Implement user preference learning
- **Collaborative Filtering**: Add user-based recommendations
- **Course Ratings**: Integrate user ratings and reviews
- **Multi-language Support**: Support for non-English course content
- **Real-time Updates**: Dynamic index updates for new courses


---
