import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Content-based recommendation system for books.
    Uses TF-IDF and cosine similarity to find similar books based on text features.
    """
    
    def __init__(self, books_df):
        """
        Initialize the content-based recommender with a books dataframe.
        
        Parameters:
        -----------
        books_df : pandas.DataFrame
            DataFrame containing book information with at least the columns:
            ISBN, Book-Title, Book-Author, Publisher
        """
        self.books_df = books_df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
    def fit(self, content_column='content'):
        """
        Fit the recommender model using TF-IDF vectorization.
        
        Parameters:
        -----------
        content_column : str
            Name of the column containing the text content to use for recommendations
        """
        # Check if content column exists, if not create it
        if content_column not in self.books_df.columns:
            self.books_df['content'] = (
                self.books_df['Book-Title'] + ' ' + 
                self.books_df['Book-Author'] + ' ' + 
                self.books_df['Publisher']
            )
            content_column = 'content'
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the content strings
        self.tfidf_matrix = tfidf.fit_transform(self.books_df[content_column])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create a Series with ISBN as index and position as value
        self.indices = pd.Series(self.books_df.index, index=self.books_df['ISBN'])
        
        return self
    
    def get_recommendations(self, book_isbn, n=10):
        """
        Get book recommendations based on similarity to the given book.
        
        Parameters:
        -----------
        book_isbn : str
            ISBN of the book to get recommendations for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Check if the book exists in our dataset
        if book_isbn not in self.indices.index:
            print(f"Book with ISBN {book_isbn} not found in the dataset.")
            return pd.DataFrame()
        
        # Get the index of the book
        idx = self.indices[book_isbn]
        
        # Get similarity scores for all books
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort books by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n most similar books (excluding the input book)
        sim_scores = sim_scores[1:n+1]
        
        # Get book indices
        book_indices = [i[0] for i in sim_scores]
        
        # Return the top n similar books
        return self.books_df.iloc[book_indices].copy()
    
    def get_recommendations_by_title(self, title, n=10):
        """
        Get book recommendations based on a book title.
        
        Parameters:
        -----------
        title : str
            Title of the book to get recommendations for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Find books with matching titles (case insensitive)
        matching_books = self.books_df[self.books_df['Book-Title'].str.lower() == title.lower()]
        
        if matching_books.empty:
            # Try partial matching if exact match fails
            matching_books = self.books_df[self.books_df['Book-Title'].str.lower().str.contains(title.lower())]
            
            if matching_books.empty:
                print(f"No books found with title containing '{title}'.")
                return pd.DataFrame()
            
            print(f"Found {len(matching_books)} books with titles containing '{title}'.")
        
        # Use the first matching book
        book_isbn = matching_books.iloc[0]['ISBN']
        
        # Get recommendations based on the ISBN
        recommendations = self.get_recommendations(book_isbn, n)
        
        return recommendations
    
    def get_recommendations_by_author(self, author, n=10):
        """
        Get book recommendations for a specific author.
        
        Parameters:
        -----------
        author : str
            Author name to get recommendations for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Find books by the author (case insensitive)
        author_books = self.books_df[self.books_df['Book-Author'].str.lower() == author.lower()]
        
        if author_books.empty:
            # Try partial matching if exact match fails
            author_books = self.books_df[self.books_df['Book-Author'].str.lower().str.contains(author.lower())]
            
            if author_books.empty:
                print(f"No books found by author containing '{author}'.")
                return pd.DataFrame()
            
            print(f"Found {len(author_books)} books by authors containing '{author}'.")
        
        # If there are multiple books by the author, use the one with the most similar books
        if len(author_books) > 1:
            # Get a sample of the author's books
            sample_size = min(3, len(author_books))
            sampled_books = author_books.sample(sample_size)
            
            # Find recommendations for each book and combine
            all_recommendations = pd.DataFrame()
            
            for _, book in sampled_books.iterrows():
                book_isbn = book['ISBN']
                recommendations = self.get_recommendations(book_isbn, n=n//sample_size)
                all_recommendations = pd.concat([all_recommendations, recommendations])
            
            # Remove duplicates and return top n
            all_recommendations = all_recommendations.drop_duplicates(subset=['ISBN'])
            return all_recommendations.head(n)
        else:
            # If there's only one book, use it for recommendations
            book_isbn = author_books.iloc[0]['ISBN']
            return self.get_recommendations(book_isbn, n)
    
    def get_similar_books_hybrid(self, book_isbn, n=10, include_same_author=True):
        """
        Get book recommendations with a hybrid approach that considers both content similarity
        and author similarity.
        
        Parameters:
        -----------
        book_isbn : str
            ISBN of the book to get recommendations for
        n : int
            Number of recommendations to return
        include_same_author : bool
            Whether to include books by the same author
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Check if the book exists in our dataset
        if book_isbn not in self.indices.index:
            print(f"Book with ISBN {book_isbn} not found in the dataset.")
            return pd.DataFrame()
        
        # Get the book details
        book_idx = self.indices[book_isbn]
        book_details = self.books_df.iloc[book_idx]
        book_author = book_details['Book-Author']
        
        # Get similarity scores for all books
        sim_scores = list(enumerate(self.cosine_sim[book_idx]))
        
        # Sort books by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 2*n most similar books (excluding the input book)
        sim_scores = sim_scores[1:2*n+1]
        
        # Get book indices
        book_indices = [i[0] for i in sim_scores]
        
        # Get the similar books
        similar_books = self.books_df.iloc[book_indices].copy()
        
        # Filter out books by the same author if requested
        if not include_same_author:
            similar_books = similar_books[similar_books['Book-Author'] != book_author]
        
        # Return the top n books
        return similar_books.head(n)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(
        books_path="../Books.csv",
        ratings_path="../Ratings.csv",
        users_path="../Users.csv"
    )
    preprocessor.load_data().clean_books_data()
    
    # Initialize and fit the content-based recommender
    recommender = ContentBasedRecommender(preprocessor.books_processed)
    recommender.fit()
    
    # Get recommendations for a book
    test_isbn = preprocessor.books_processed['ISBN'].iloc[0]
    recommendations = recommender.get_recommendations(test_isbn, n=5)
    
    print(f"Recommendations for book: {preprocessor.books_processed.loc[preprocessor.books_processed['ISBN'] == test_isbn, 'Book-Title'].values[0]}")
    print(recommendations[['Book-Title', 'Book-Author']])
