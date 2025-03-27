import pandas as pd
import numpy as np

class PopularityRecommender:
    """
    Popularity-based recommendation system for books.
    Recommends books based on popularity metrics like number of ratings and average rating.
    """
    
    def __init__(self, ratings_df, books_df):
        """
        Initialize the popularity-based recommender.
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings with columns: User-ID, ISBN, Book-Rating
        books_df : pandas.DataFrame
            DataFrame containing book information with at least the columns:
            ISBN, Book-Title, Book-Author
        """
        self.ratings_df = ratings_df
        self.books_df = books_df
        self.popularity_df = None
        
    def fit(self, min_ratings=10):
        """
        Calculate popularity metrics for all books.
        
        Parameters:
        -----------
        min_ratings : int
            Minimum number of ratings a book must have to be considered
        """
        print("Calculating popularity metrics...")
        
        # Group by ISBN and calculate statistics
        book_stats = self.ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['count', 'mean']
        })
        
        # Flatten the column names
        book_stats.columns = ['rating_count', 'rating_mean']
        book_stats = book_stats.reset_index()
        
        # Filter books with minimum number of ratings
        book_stats = book_stats[book_stats['rating_count'] >= min_ratings]
        
        # Calculate a popularity score
        # This is a weighted score that considers both the number of ratings and the average rating
        C = book_stats['rating_mean'].mean()  # Mean rating across all books
        m = min_ratings  # Minimum ratings required
        
        def weighted_rating(x):
            v = x['rating_count']
            R = x['rating_mean']
            return (v / (v + m) * R) + (m / (v + m) * C)
        
        book_stats['popularity_score'] = book_stats.apply(weighted_rating, axis=1)
        
        # Merge with book details
        self.popularity_df = pd.merge(book_stats, self.books_df, on='ISBN')
        
        print(f"Calculated popularity metrics for {len(self.popularity_df)} books.")
        return self
    
    def recommend(self, n=10, criteria='popularity_score'):
        """
        Get the most popular books based on the specified criteria.
        
        Parameters:
        -----------
        n : int
            Number of recommendations to return
        criteria : str
            Criteria to sort by: 'popularity_score', 'rating_count', or 'rating_mean'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        if self.popularity_df is None:
            print("Error: You must call fit() before recommend().")
            return pd.DataFrame()
        
        # Sort by the specified criteria
        if criteria == 'rating_count':
            sorted_df = self.popularity_df.sort_values('rating_count', ascending=False)
        elif criteria == 'rating_mean':
            sorted_df = self.popularity_df.sort_values('rating_mean', ascending=False)
        else:  # Default to popularity_score
            sorted_df = self.popularity_df.sort_values('popularity_score', ascending=False)
        
        # Return top n recommendations
        return sorted_df.head(n)
    
    def recommend_by_year(self, year, n=10, criteria='popularity_score'):
        """
        Get the most popular books for a specific publication year.
        
        Parameters:
        -----------
        year : int
            Publication year to filter by
        n : int
            Number of recommendations to return
        criteria : str
            Criteria to sort by: 'popularity_score', 'rating_count', or 'rating_mean'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        if self.popularity_df is None:
            print("Error: You must call fit() before recommend_by_year().")
            return pd.DataFrame()
        
        # Filter by year
        year_df = self.popularity_df[self.popularity_df['Year-Of-Publication'] == year]
        
        if year_df.empty:
            print(f"No books found for year {year}.")
            return pd.DataFrame()
        
        # Sort by the specified criteria
        if criteria == 'rating_count':
            sorted_df = year_df.sort_values('rating_count', ascending=False)
        elif criteria == 'rating_mean':
            sorted_df = year_df.sort_values('rating_mean', ascending=False)
        else:  # Default to popularity_score
            sorted_df = year_df.sort_values('popularity_score', ascending=False)
        
        # Return top n recommendations
        return sorted_df.head(n)
    
    def recommend_by_publisher(self, publisher, n=10, criteria='popularity_score'):
        """
        Get the most popular books for a specific publisher.
        
        Parameters:
        -----------
        publisher : str
            Publisher to filter by (case insensitive, partial match)
        n : int
            Number of recommendations to return
        criteria : str
            Criteria to sort by: 'popularity_score', 'rating_count', or 'rating_mean'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        if self.popularity_df is None:
            print("Error: You must call fit() before recommend_by_publisher().")
            return pd.DataFrame()
        
        # Filter by publisher (case insensitive, partial match)
        publisher_df = self.popularity_df[
            self.popularity_df['Publisher'].str.lower().str.contains(publisher.lower())
        ]
        
        if publisher_df.empty:
            print(f"No books found for publisher containing '{publisher}'.")
            return pd.DataFrame()
        
        # Sort by the specified criteria
        if criteria == 'rating_count':
            sorted_df = publisher_df.sort_values('rating_count', ascending=False)
        elif criteria == 'rating_mean':
            sorted_df = publisher_df.sort_values('rating_mean', ascending=False)
        else:  # Default to popularity_score
            sorted_df = publisher_df.sort_values('popularity_score', ascending=False)
        
        # Return top n recommendations
        return sorted_df.head(n)
    
    def get_trending_by_decade(self, n=5):
        """
        Get the most popular books for each decade.
        
        Parameters:
        -----------
        n : int
            Number of books to return per decade
            
        Returns:
        --------
        dict
            Dictionary with decades as keys and DataFrames of popular books as values
        """
        if self.popularity_df is None:
            print("Error: You must call fit() before get_trending_by_decade().")
            return {}
        
        # Create a decade column
        self.popularity_df['decade'] = (self.popularity_df['Year-Of-Publication'] // 10) * 10
        
        # Filter out invalid years
        valid_decades = self.popularity_df[
            (self.popularity_df['decade'] >= 1900) & 
            (self.popularity_df['decade'] <= 2020)
        ]
        
        # Group by decade and get top n books for each
        trending_by_decade = {}
        
        for decade, group in valid_decades.groupby('decade'):
            # Sort by popularity score
            sorted_group = group.sort_values('popularity_score', ascending=False)
            trending_by_decade[decade] = sorted_group.head(n)
        
        return trending_by_decade


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
    preprocessor.process_all(min_book_ratings=10, min_user_ratings=5)
    
    # Initialize and fit the popularity recommender
    recommender = PopularityRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    recommender.fit(min_ratings=20)
    
    # Get overall popular books
    popular_books = recommender.recommend(n=5)
    print("Top 5 Popular Books:")
    print(popular_books[['Book-Title', 'Book-Author', 'rating_count', 'rating_mean', 'popularity_score']])
    
    # Get popular books by year
    year_books = recommender.recommend_by_year(2000, n=5)
    print("\nTop 5 Popular Books from 2000:")
    print(year_books[['Book-Title', 'Book-Author', 'rating_count', 'rating_mean']])
    
    # Get trending books by decade
    trending = recommender.get_trending_by_decade(n=3)
    for decade, books in trending.items():
        print(f"\nTop 3 Books from {decade}s:")
        print(books[['Book-Title', 'Book-Author', 'rating_count', 'rating_mean']])
