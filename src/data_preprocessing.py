import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataPreprocessor:
    def __init__(self, books_path, ratings_path, users_path):
        """
        Initialize the DataPreprocessor with paths to the dataset files.
        
        Parameters:
        -----------
        books_path : str
            Path to the Books.csv file
        ratings_path : str
            Path to the Ratings.csv file
        users_path : str
            Path to the Users.csv file
        """
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.users_path = users_path
        
        # Dataframes
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        
        # Processed dataframes
        self.books_processed = None
        self.ratings_processed = None
        self.users_processed = None
        
        # Combined data
        self.books_with_ratings = None
        
    def load_data(self):
        """Load the dataset files into pandas dataframes."""
        print("Loading data...")
        
        # Load books data
        self.books_df = pd.read_csv(self.books_path, encoding='latin-1', 
                                    on_bad_lines='skip', low_memory=False)
        
        # Load ratings data
        self.ratings_df = pd.read_csv(self.ratings_path, encoding='latin-1', 
                                     on_bad_lines='skip', low_memory=False)
        
        # Load users data
        self.users_df = pd.read_csv(self.users_path, encoding='latin-1', 
                                   on_bad_lines='skip', low_memory=False)
        
        print(f"Loaded {len(self.books_df)} books, {len(self.ratings_df)} ratings, and {len(self.users_df)} users.")
        return self
    
    def clean_books_data(self):
        """Clean and preprocess the books dataframe."""
        print("Cleaning books data...")
        
        # Create a copy to avoid modifying the original
        self.books_processed = self.books_df.copy()
        
        # Drop duplicates based on ISBN
        self.books_processed.drop_duplicates(subset=['ISBN'], inplace=True)
        
        # Fill missing values
        self.books_processed['Book-Title'] = self.books_processed['Book-Title'].fillna('Unknown Title')
        self.books_processed['Book-Author'] = self.books_processed['Book-Author'].fillna('Unknown Author')
        self.books_processed['Publisher'] = self.books_processed['Publisher'].fillna('Unknown Publisher')
        self.books_processed['Year-Of-Publication'] = self.books_processed['Year-Of-Publication'].fillna('0')
        
        # Convert year to numeric, coercing errors to NaN, then fill NaNs with 0
        self.books_processed['Year-Of-Publication'] = pd.to_numeric(
            self.books_processed['Year-Of-Publication'], errors='coerce').fillna(0).astype(int)
        
        # Create a content string for content-based filtering
        self.books_processed['content'] = (
            self.books_processed['Book-Title'] + ' ' + 
            self.books_processed['Book-Author'] + ' ' + 
            self.books_processed['Publisher']
        )
        
        print(f"Cleaned books data. {len(self.books_processed)} books after cleaning.")
        return self
    
    def clean_ratings_data(self, min_book_ratings=10, min_user_ratings=10):
        """
        Clean and preprocess the ratings dataframe.
        
        Parameters:
        -----------
        min_book_ratings : int
            Minimum number of ratings a book must have to be included
        min_user_ratings : int
            Minimum number of ratings a user must have given to be included
        """
        print("Cleaning ratings data...")
        
        # Create a copy to avoid modifying the original
        self.ratings_processed = self.ratings_df.copy()
        
        # Convert ratings to numeric
        self.ratings_processed['Book-Rating'] = pd.to_numeric(
            self.ratings_processed['Book-Rating'], errors='coerce')
        
        # Remove rows with missing ratings
        self.ratings_processed = self.ratings_processed.dropna(subset=['Book-Rating'])
        
        # Filter out implicit ratings (ratings of 0)
        self.ratings_processed = self.ratings_processed[self.ratings_processed['Book-Rating'] > 0]
        
        # Filter based on number of ratings per book and per user
        book_counts = self.ratings_processed['ISBN'].value_counts()
        user_counts = self.ratings_processed['User-ID'].value_counts()
        
        popular_books = book_counts[book_counts >= min_book_ratings].index
        active_users = user_counts[user_counts >= min_user_ratings].index
        
        self.ratings_processed = self.ratings_processed[
            self.ratings_processed['ISBN'].isin(popular_books) & 
            self.ratings_processed['User-ID'].isin(active_users)
        ]
        
        print(f"Cleaned ratings data. {len(self.ratings_processed)} ratings after cleaning.")
        return self
    
    def clean_users_data(self):
        """Clean and preprocess the users dataframe."""
        print("Cleaning users data...")
        
        # Create a copy to avoid modifying the original
        self.users_processed = self.users_df.copy()
        
        # Convert age to numeric, coercing errors to NaN
        self.users_processed['Age'] = pd.to_numeric(self.users_processed['Age'], errors='coerce')
        
        # Filter out unreasonable ages (e.g., too young or too old)
        self.users_processed = self.users_processed[
            (self.users_processed['Age'].isna()) | 
            ((self.users_processed['Age'] >= 5) & (self.users_processed['Age'] <= 100))
        ]
        
        # Extract country from location
        def extract_country(location):
            if pd.isna(location):
                return 'Unknown'
            parts = location.split(',')
            if len(parts) >= 3:
                return parts[2].strip()
            elif len(parts) == 2:
                return parts[1].strip()
            else:
                return parts[0].strip()
        
        self.users_processed['Country'] = self.users_processed['Location'].apply(extract_country)
        
        print(f"Cleaned users data. {len(self.users_processed)} users after cleaning.")
        return self
    
    def merge_data(self):
        """Merge the processed dataframes."""
        print("Merging data...")
        
        # Merge books and ratings
        self.books_with_ratings = pd.merge(
            self.ratings_processed, 
            self.books_processed,
            on='ISBN', 
            how='inner'
        )
        
        # Merge with users if needed
        if self.users_processed is not None:
            self.books_with_ratings = pd.merge(
                self.books_with_ratings,
                self.users_processed,
                on='User-ID',
                how='left'
            )
        
        print(f"Merged data. Final dataset has {len(self.books_with_ratings)} entries.")
        return self
    
    def create_user_item_matrix(self):
        """Create a user-item matrix for collaborative filtering."""
        print("Creating user-item matrix...")
        
        # Create the user-item matrix
        user_item_matrix = self.ratings_processed.pivot(
            index='User-ID', 
            columns='ISBN', 
            values='Book-Rating'
        ).fillna(0)
        
        print(f"Created user-item matrix with shape {user_item_matrix.shape}")
        return user_item_matrix
    
    def create_item_feature_matrix(self):
        """Create an item feature matrix for content-based filtering."""
        print("Creating item feature matrix...")
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the content strings
        tfidf_matrix = tfidf.fit_transform(self.books_processed['content'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a DataFrame with ISBN as index
        indices = pd.Series(self.books_processed.index, index=self.books_processed['ISBN'])
        
        print(f"Created item feature matrix with shape {tfidf_matrix.shape}")
        return cosine_sim, indices
    
    def get_popular_books(self, n=50):
        """Get the most popular books based on number of ratings."""
        book_ratings_count = self.ratings_processed['ISBN'].value_counts()
        popular_books = book_ratings_count.nlargest(n).index
        
        popular_books_df = self.books_processed[self.books_processed['ISBN'].isin(popular_books)]
        return popular_books_df
    
    def get_top_rated_books(self, min_ratings=10, n=50):
        """Get the top-rated books with a minimum number of ratings."""
        # Group by ISBN and calculate mean rating and count
        book_stats = self.ratings_processed.groupby('ISBN').agg({
            'Book-Rating': ['mean', 'count']
        })
        
        book_stats.columns = ['avg_rating', 'num_ratings']
        book_stats = book_stats.reset_index()
        
        # Filter books with minimum number of ratings
        qualified_books = book_stats[book_stats['num_ratings'] >= min_ratings]
        
        # Sort by average rating and get top n
        top_rated = qualified_books.sort_values('avg_rating', ascending=False).head(n)
        
        # Merge with book details
        top_rated_books = pd.merge(top_rated, self.books_processed, on='ISBN')
        
        return top_rated_books
    
    def process_all(self, min_book_ratings=10, min_user_ratings=10):
        """Run all preprocessing steps."""
        return (self.load_data()
                .clean_books_data()
                .clean_ratings_data(min_book_ratings, min_user_ratings)
                .clean_users_data()
                .merge_data())

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        books_path="../Books.csv",
        ratings_path="../Ratings.csv",
        users_path="../Users.csv"
    )
    
    # Process all data
    preprocessor.process_all(min_book_ratings=10, min_user_ratings=5)
    
    # Get popular books
    popular_books = preprocessor.get_popular_books(n=10)
    print("\nTop 10 Popular Books:")
    print(popular_books[['ISBN', 'Book-Title', 'Book-Author']].head(10))
    
    # Get top-rated books
    top_rated = preprocessor.get_top_rated_books(min_ratings=20, n=10)
    print("\nTop 10 Highly Rated Books:")
    print(top_rated[['ISBN', 'Book-Title', 'Book-Author', 'avg_rating', 'num_ratings']].head(10))
