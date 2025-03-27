import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Import our modules
from src.data_preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.popularity_based import PopularityRecommender

# Set page configuration
st.set_page_config(
    page_title="BookBud - Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .book-title {
        font-weight: bold;
        color: #1E3A8A;
    }
    .book-author {
        font-style: italic;
        color: #4B5563;
    }
    .book-rating {
        color: #F59E0B;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Cache the data loading and preprocessing
@st.cache_data
def load_and_preprocess_data(min_book_ratings=10, min_user_ratings=5):
    """Load and preprocess the data."""
    with st.spinner("Loading and preprocessing data... This may take a few minutes."):
        preprocessor = DataPreprocessor(
            books_path="Books.csv",
            ratings_path="Ratings.csv",
            users_path="Users.csv"
        )
        preprocessor.process_all(min_book_ratings, min_user_ratings)
        return preprocessor

# Cache the recommender initialization
@st.cache_resource
def initialize_recommenders(preprocessor):
    """Initialize the recommendation systems."""
    with st.spinner("Initializing recommendation systems..."):
        # Content-based recommender
        content_recommender = ContentBasedRecommender(preprocessor.books_processed)
        content_recommender.fit()
        
        # Collaborative filtering recommender
        collab_recommender = CollaborativeFilteringRecommender(
            preprocessor.ratings_processed, 
            preprocessor.books_processed
        )
        collab_recommender.fit()
        
        # Popularity-based recommender
        popularity_recommender = PopularityRecommender(
            preprocessor.ratings_processed, 
            preprocessor.books_processed
        )
        popularity_recommender.fit(min_ratings=20)
        
        return content_recommender, collab_recommender, popularity_recommender

def display_book_card(book, show_rating=True):
    """Display a book in a card format."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display book image if available
        if 'Image-URL-M' in book and pd.notna(book['Image-URL-M']):
            st.image(book['Image-URL-M'], width=100)
        else:
            st.image("https://via.placeholder.com/100x150?text=No+Image", width=100)
    
    with col2:
        st.markdown(f"<p class='book-title'>{book['Book-Title']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='book-author'>by {book['Book-Author']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Published: {int(book['Year-Of-Publication'])} | Publisher: {book['Publisher']}</p>", unsafe_allow_html=True)
        
        if show_rating and 'rating_mean' in book and pd.notna(book['rating_mean']):
            st.markdown(f"<p class='book-rating'>Rating: {book['rating_mean']:.2f}/10 ({int(book['rating_count'])} ratings)</p>", unsafe_allow_html=True)
        
        st.markdown(f"ISBN: {book['ISBN']}")

def display_books_grid(books_df, show_rating=True, cols=2):
    """Display a grid of books."""
    if books_df.empty:
        st.info("No books found matching the criteria.")
        return
    
    # Create rows with the specified number of columns
    for i in range(0, len(books_df), cols):
        row_books = books_df.iloc[i:min(i+cols, len(books_df))]
        columns = st.columns(cols)
        
        for j, (_, book) in enumerate(row_books.iterrows()):
            with columns[j]:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    display_book_card(book, show_rating)
                    st.markdown("</div>", unsafe_allow_html=True)

def display_data_insights(preprocessor):
    """Display insights about the dataset."""
    st.markdown("<h2 class='sub-header'>Dataset Insights</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Books", f"{len(preprocessor.books_processed):,}")
    
    with col2:
        st.metric("Total Ratings", f"{len(preprocessor.ratings_processed):,}")
    
    with col3:
        st.metric("Total Users", f"{len(preprocessor.users_processed):,}")
    
    # Books published by year
    st.markdown("<h3 class='sub-header'>Books Published by Year</h3>", unsafe_allow_html=True)
    
    # Filter out unreasonable years
    year_counts = preprocessor.books_processed[
        (preprocessor.books_processed['Year-Of-Publication'] >= 1900) & 
        (preprocessor.books_processed['Year-Of-Publication'] <= 2023)
    ]['Year-Of-Publication'].value_counts().sort_index()
    
    fig = px.bar(
        x=year_counts.index, 
        y=year_counts.values,
        labels={'x': 'Year', 'y': 'Number of Books'},
        title='Books Published by Year'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top publishers
    st.markdown("<h3 class='sub-header'>Top Publishers</h3>", unsafe_allow_html=True)
    
    publisher_counts = preprocessor.books_processed['Publisher'].value_counts().head(10)
    fig = px.bar(
        x=publisher_counts.values, 
        y=publisher_counts.index,
        orientation='h',
        labels={'x': 'Number of Books', 'y': 'Publisher'},
        title='Top 10 Publishers'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating distribution
    st.markdown("<h3 class='sub-header'>Rating Distribution</h3>", unsafe_allow_html=True)
    
    rating_counts = preprocessor.ratings_processed['Book-Rating'].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index, 
        y=rating_counts.values,
        labels={'x': 'Rating', 'y': 'Count'},
        title='Distribution of Ratings'
    )
    st.plotly_chart(fig, use_container_width=True)

def home_page(preprocessor, popularity_recommender):
    """Display the home page."""
    st.markdown("<h1 class='main-header'>📚 BookBud</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>A comprehensive book recommendation system</p>", unsafe_allow_html=True)
    
    # Get popular books
    popular_books = popularity_recommender.recommend(n=6)
    
    st.markdown("<h2 class='sub-header'>Popular Books</h2>", unsafe_allow_html=True)
    display_books_grid(popular_books, cols=3)
    
    # Display dataset insights
    display_data_insights(preprocessor)

def content_based_page(content_recommender, preprocessor):
    """Display the content-based recommendation page."""
    st.markdown("<h1 class='main-header'>Content-Based Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Get recommendations based on book content, authors, and publishers</p>", unsafe_allow_html=True)
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs(["By Title", "By Author", "By ISBN"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Find Books Similar to a Title</h2>", unsafe_allow_html=True)
        
        # Get a list of book titles for the dropdown
        book_titles = preprocessor.books_processed['Book-Title'].unique()
        book_titles_sorted = sorted(book_titles)
        
        # Create a search box with autocomplete
        title_search = st.text_input("Enter a book title", key="title_search")
        
        if title_search:
            # Filter titles based on search
            matching_titles = [title for title in book_titles_sorted 
                              if title_search.lower() in title.lower()]
            
            if matching_titles:
                selected_title = st.selectbox("Select a book", matching_titles)
                
                if st.button("Get Recommendations", key="title_rec_button"):
                    with st.spinner("Finding similar books..."):
                        recommendations = content_recommender.get_recommendations_by_title(selected_title, n=6)
                        
                        if not recommendations.empty:
                            st.markdown("<h3 class='sub-header'>Recommended Books</h3>", unsafe_allow_html=True)
                            display_books_grid(recommendations, show_rating=False, cols=3)
                        else:
                            st.info("No recommendations found for this book.")
            else:
                st.info("No books found matching your search.")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Find Books by Author</h2>", unsafe_allow_html=True)
        
        # Get a list of authors for the dropdown
        authors = preprocessor.books_processed['Book-Author'].unique()
        authors_sorted = sorted(authors)
        
        # Create a search box with autocomplete
        author_search = st.text_input("Enter an author name", key="author_search")
        
        if author_search:
            # Filter authors based on search
            matching_authors = [author for author in authors_sorted 
                               if author_search.lower() in author.lower()]
            
            if matching_authors:
                selected_author = st.selectbox("Select an author", matching_authors)
                
                if st.button("Get Recommendations", key="author_rec_button"):
                    with st.spinner("Finding books by this author..."):
                        recommendations = content_recommender.get_recommendations_by_author(selected_author, n=6)
                        
                        if not recommendations.empty:
                            st.markdown("<h3 class='sub-header'>Recommended Books</h3>", unsafe_allow_html=True)
                            display_books_grid(recommendations, show_rating=False, cols=3)
                        else:
                            st.info("No recommendations found for this author.")
            else:
                st.info("No authors found matching your search.")
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Find Books by ISBN</h2>", unsafe_allow_html=True)
        
        isbn_input = st.text_input("Enter an ISBN", key="isbn_input")
        
        if isbn_input:
            if st.button("Get Recommendations", key="isbn_rec_button"):
                with st.spinner("Finding similar books..."):
                    # Check if ISBN exists in the dataset
                    if isbn_input in preprocessor.books_processed['ISBN'].values:
                        # Get the book details
                        book = preprocessor.books_processed[preprocessor.books_processed['ISBN'] == isbn_input].iloc[0]
                        
                        st.markdown("<h3 class='sub-header'>Selected Book</h3>", unsafe_allow_html=True)
                        with st.container():
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            display_book_card(book, show_rating=False)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Get recommendations
                        recommendations = content_recommender.get_recommendations(isbn_input, n=6)
                        
                        if not recommendations.empty:
                            st.markdown("<h3 class='sub-header'>Recommended Books</h3>", unsafe_allow_html=True)
                            display_books_grid(recommendations, show_rating=False, cols=3)
                        else:
                            st.info("No recommendations found for this book.")
                    else:
                        st.error("ISBN not found in the dataset.")

def collaborative_filtering_page(collab_recommender, preprocessor):
    """Display the collaborative filtering recommendation page."""
    st.markdown("<h1 class='main-header'>Collaborative Filtering Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Get recommendations based on user ratings and behavior</p>", unsafe_allow_html=True)
    
    # Create tabs for different recommendation methods
    tab1, tab2 = st.tabs(["User-Based", "Item-Based"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>User-Based Recommendations</h2>", unsafe_allow_html=True)
        st.markdown("Get recommendations based on similar users' preferences.")
        
        # Get a list of user IDs for the dropdown
        user_ids = preprocessor.ratings_processed['User-ID'].unique()
        
        # Create a search box
        user_id_input = st.text_input("Enter a User ID", key="user_id_input")
        
        if user_id_input:
            try:
                user_id = int(user_id_input)
                
                if user_id in user_ids:
                    if st.button("Get Recommendations", key="user_rec_button"):
                        with st.spinner("Finding recommendations for this user..."):
                            recommendations = collab_recommender.user_based_recommendations(user_id, n=6)
                            
                            if not recommendations.empty:
                                st.markdown("<h3 class='sub-header'>Recommended Books</h3>", unsafe_allow_html=True)
                                display_books_grid(recommendations, show_rating=False, cols=3)
                            else:
                                st.info("No recommendations found for this user.")
                else:
                    st.error("User ID not found in the dataset.")
            except ValueError:
                st.error("Please enter a valid User ID (integer).")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Item-Based Recommendations</h2>", unsafe_allow_html=True)
        st.markdown("Get recommendations for similar books based on user ratings.")
        
        # Create a search box for ISBN
        isbn_input = st.text_input("Enter an ISBN", key="isbn_input_collab")
        
        if isbn_input:
            if st.button("Get Recommendations", key="item_rec_button"):
                with st.spinner("Finding similar books..."):
                    # Check if ISBN exists in the dataset
                    if isbn_input in preprocessor.books_processed['ISBN'].values:
                        # Get the book details
                        book = preprocessor.books_processed[preprocessor.books_processed['ISBN'] == isbn_input].iloc[0]
                        
                        st.markdown("<h3 class='sub-header'>Selected Book</h3>", unsafe_allow_html=True)
                        with st.container():
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            display_book_card(book, show_rating=False)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Get recommendations
                        recommendations = collab_recommender.get_recommendations_for_book(isbn_input, n=6)
                        
                        if not recommendations.empty:
                            st.markdown("<h3 class='sub-header'>Recommended Books</h3>", unsafe_allow_html=True)
                            display_books_grid(recommendations, show_rating=False, cols=3)
                        else:
                            st.info("No recommendations found for this book.")
                    else:
                        st.error("ISBN not found in the dataset.")

def popularity_based_page(popularity_recommender):
    """Display the popularity-based recommendation page."""
    st.markdown("<h1 class='main-header'>Popularity-Based Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Discover popular books based on user ratings</p>", unsafe_allow_html=True)
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs(["Overall Popular", "By Year", "By Publisher"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Overall Popular Books</h2>", unsafe_allow_html=True)
        
        # Sorting criteria
        sort_by = st.selectbox(
            "Sort by",
            ["Popularity Score", "Number of Ratings", "Average Rating"],
            key="sort_overall"
        )
        
        # Map selection to criteria
        criteria_map = {
            "Popularity Score": "popularity_score",
            "Number of Ratings": "rating_count",
            "Average Rating": "rating_mean"
        }
        
        # Number of books to show
        num_books = st.slider("Number of books to show", 5, 20, 10, key="num_overall")
        
        if st.button("Get Popular Books", key="popular_button"):
            with st.spinner("Finding popular books..."):
                popular_books = popularity_recommender.recommend(
                    n=num_books,
                    criteria=criteria_map[sort_by]
                )
                
                if not popular_books.empty:
                    st.markdown("<h3 class='sub-header'>Popular Books</h3>", unsafe_allow_html=True)
                    display_books_grid(popular_books, cols=2)
                else:
                    st.info("No books found.")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Popular Books by Year</h2>", unsafe_allow_html=True)
        
        # Year selection
        min_year = 1900
        max_year = 2023
        year = st.slider("Select a year", min_year, max_year, 2000, key="year_slider")
        
        # Sorting criteria
        sort_by = st.selectbox(
            "Sort by",
            ["Popularity Score", "Number of Ratings", "Average Rating"],
            key="sort_year"
        )
        
        # Number of books to show
        num_books = st.slider("Number of books to show", 5, 20, 10, key="num_year")
        
        if st.button("Get Popular Books for Year", key="year_button"):
            with st.spinner(f"Finding popular books from {year}..."):
                year_books = popularity_recommender.recommend_by_year(
                    year=year,
                    n=num_books,
                    criteria=criteria_map[sort_by]
                )
                
                if not year_books.empty:
                    st.markdown(f"<h3 class='sub-header'>Popular Books from {year}</h3>", unsafe_allow_html=True)
                    display_books_grid(year_books, cols=2)
                else:
                    st.info(f"No books found for the year {year}.")
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Popular Books by Publisher</h2>", unsafe_allow_html=True)
        
        # Publisher input
        publisher_input = st.text_input("Enter a publisher name", key="publisher_input")
        
        # Sorting criteria
        sort_by = st.selectbox(
            "Sort by",
            ["Popularity Score", "Number of Ratings", "Average Rating"],
            key="sort_publisher"
        )
        
        # Number of books to show
        num_books = st.slider("Number of books to show", 5, 20, 10, key="num_publisher")
        
        if publisher_input:
            if st.button("Get Popular Books by Publisher", key="publisher_button"):
                with st.spinner(f"Finding popular books from {publisher_input}..."):
                    publisher_books = popularity_recommender.recommend_by_publisher(
                        publisher=publisher_input,
                        n=num_books,
                        criteria=criteria_map[sort_by]
                    )
                    
                    if not publisher_books.empty:
                        st.markdown(f"<h3 class='sub-header'>Popular Books from {publisher_input}</h3>", unsafe_allow_html=True)
                        display_books_grid(publisher_books, cols=2)
                    else:
                        st.info(f"No books found for publisher '{publisher_input}'.")

def about_page():
    """Display the about page."""
    st.markdown("<h1 class='main-header'>About BookBud</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    BookBud is a comprehensive book recommendation system built using the Book-Crossing dataset. 
    This project demonstrates various recommendation techniques and provides a user-friendly web interface for exploring book recommendations.
    
    ## Dataset
    
    The project uses the Book-Crossing dataset which contains:
    - **Books.csv**: Information about books (ISBN, title, author, year, publisher, etc.)
    - **Ratings.csv**: User ratings for books
    - **Users.csv**: User demographic information
    
    ## Recommendation Algorithms
    
    BookBud implements three main recommendation approaches:
    
    ### 1. Content-Based Filtering
    Recommends books similar to a book you like based on book attributes like title, author, and publisher.
    
    ### 2. Collaborative Filtering
    - **User-Based**: Recommends books that similar users have liked
    - **Item-Based**: Recommends books similar to books you've liked based on user rating patterns
    
    ### 3. Popularity-Based
    Recommends the most popular books based on number of ratings and average rating.
    
    ## Technologies Used
    
    - Python
    - Pandas for data manipulation
    - Scikit-learn for machine learning algorithms
    - Streamlit for the web interface
    - Plotly for interactive visualizations
    
    ## Future Improvements
    
    - Implement hybrid recommendation algorithms
    - Add user authentication and personalized recommendations
    - Incorporate more recent book data
    - Add natural language processing for book descriptions
    """)

def main():
    """Main function to run the Streamlit app."""
    # Load and preprocess data
    preprocessor = load_and_preprocess_data(min_book_ratings=10, min_user_ratings=5)
    
    # Initialize recommenders
    content_recommender, collab_recommender, popularity_recommender = initialize_recommenders(preprocessor)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Content-Based Recommendations", "Collaborative Filtering", "Popularity-Based", "About"]
    )
    
    # Display the selected page
    if page == "Home":
        home_page(preprocessor, popularity_recommender)
    elif page == "Content-Based Recommendations":
        content_based_page(content_recommender, preprocessor)
    elif page == "Collaborative Filtering":
        collaborative_filtering_page(collab_recommender, preprocessor)
    elif page == "Popularity-Based":
        popularity_based_page(popularity_recommender)
    elif page == "About":
        about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("BookBud - Book Recommendation System")
    st.sidebar.text("© 2025")

if __name__ == "__main__":
    main()
