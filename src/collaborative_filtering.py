import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommendation system for books.
    Implements both user-based and item-based collaborative filtering.
    """
    
    def __init__(self, ratings_df, books_df):
        """
        Initialize the collaborative filtering recommender.
        
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
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_means = None
        
    def create_matrices(self):
        """Create user-item and item-user matrices for collaborative filtering."""
        print("Creating user-item matrix...")
        
        # Create the user-item matrix
        self.user_item_matrix = self.ratings_df.pivot(
            index='User-ID', 
            columns='ISBN', 
            values='Book-Rating'
        ).fillna(0)
        
        # Create the item-user matrix (transpose of user-item matrix)
        self.item_user_matrix = self.user_item_matrix.T
        
        print(f"Created user-item matrix with shape {self.user_item_matrix.shape}")
        return self
    
    def compute_user_similarity(self):
        """Compute user-user similarity matrix."""
        print("Computing user similarity...")
        
        # Convert to sparse matrix for efficiency
        user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(user_item_sparse)
        
        # Create DataFrame with user IDs as index and columns
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"Computed user similarity matrix with shape {self.user_similarity.shape}")
        return self
    
    def compute_item_similarity(self):
        """Compute item-item similarity matrix."""
        print("Computing item similarity...")
        
        # Convert to sparse matrix for efficiency
        item_user_sparse = csr_matrix(self.item_user_matrix.values)
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(item_user_sparse)
        
        # Create DataFrame with ISBNs as index and columns
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index
        )
        
        print(f"Computed item similarity matrix with shape {self.item_similarity.shape}")
        return self
    
    def compute_user_means(self):
        """Compute mean ratings for each user (for user-based CF with normalization)."""
        # Replace zeros with NaN to compute mean of actual ratings
        user_item_nonzero = self.user_item_matrix.replace(0, np.nan)
        
        # Compute mean rating for each user
        self.user_means = user_item_nonzero.mean(axis=1)
        
        return self
    
    def fit(self):
        """Fit the collaborative filtering model."""
        return (self.create_matrices()
                .compute_user_similarity()
                .compute_item_similarity()
                .compute_user_means())
    
    def user_based_recommendations(self, user_id, n=10, k=20):
        """
        Generate user-based collaborative filtering recommendations.
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n : int
            Number of recommendations to return
        k : int
            Number of similar users to consider
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Check if the user exists in our dataset
        if user_id not in self.user_item_matrix.index:
            print(f"User with ID {user_id} not found in the dataset.")
            return pd.DataFrame()
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get the user's mean rating
        user_mean = self.user_means.get(user_id, 0)
        
        # Get similarity scores for all users
        user_similarities = self.user_similarity[user_id].drop(user_id)
        
        # Get top k similar users
        similar_users = user_similarities.nlargest(k)
        
        # Initialize dictionary to store predicted ratings
        predicted_ratings = {}
        
        # Get books the user hasn't rated
        unrated_books = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings for each unrated book
        for book in unrated_books:
            # Get ratings for this book from similar users
            book_ratings = self.user_item_matrix[book]
            
            # Calculate weighted average rating
            numerator = 0
            denominator = 0
            
            for sim_user, similarity in similar_users.items():
                # Get the similar user's rating for this book
                rating = book_ratings.get(sim_user, 0)
                
                # Skip if the similar user hasn't rated this book
                if rating == 0:
                    continue
                
                # Get the similar user's mean rating
                sim_user_mean = self.user_means.get(sim_user, 0)
                
                # Add to weighted average (with normalization)
                numerator += similarity * (rating - sim_user_mean)
                denominator += abs(similarity)
            
            # Calculate predicted rating if we have data
            if denominator > 0:
                predicted_ratings[book] = user_mean + (numerator / denominator)
        
        # Sort books by predicted rating
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Get top n recommendations
        top_recommendations = sorted_predictions[:n]
        
        # Get book details for recommendations
        recommended_books = []
        for isbn, _ in top_recommendations:
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                recommended_books.append(book_info.iloc[0])
        
        return pd.DataFrame(recommended_books)
    
    def item_based_recommendations(self, user_id, n=10):
        """
        Generate item-based collaborative filtering recommendations.
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Check if the user exists in our dataset
        if user_id not in self.user_item_matrix.index:
            print(f"User with ID {user_id} not found in the dataset.")
            return pd.DataFrame()
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get books the user has rated
        rated_books = user_ratings[user_ratings > 0]
        
        # Get books the user hasn't rated
        unrated_books = user_ratings[user_ratings == 0].index
        
        # Initialize dictionary to store predicted ratings
        predicted_ratings = {}
        
        # Calculate predicted ratings for each unrated book
        for unrated_book in unrated_books:
            # Initialize variables for weighted average
            weighted_sum = 0
            similarity_sum = 0
            
            # Calculate similarity-weighted average of the user's ratings
            for rated_book, rating in rated_books.items():
                # Get similarity between the rated book and the unrated book
                similarity = self.item_similarity.loc[rated_book, unrated_book]
                
                # Add to weighted sum
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            # Calculate predicted rating if we have data
            if similarity_sum > 0:
                predicted_ratings[unrated_book] = weighted_sum / similarity_sum
        
        # Sort books by predicted rating
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Get top n recommendations
        top_recommendations = sorted_predictions[:n]
        
        # Get book details for recommendations
        recommended_books = []
        for isbn, _ in top_recommendations:
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                recommended_books.append(book_info.iloc[0])
        
        return pd.DataFrame(recommended_books)
    
    def get_recommendations_for_book(self, book_isbn, n=10):
        """
        Get similar books based on collaborative filtering.
        
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
        if book_isbn not in self.item_similarity.index:
            print(f"Book with ISBN {book_isbn} not found in the dataset.")
            return pd.DataFrame()
        
        # Get similarity scores for all books
        book_similarities = self.item_similarity[book_isbn].drop(book_isbn)
        
        # Get top n similar books
        similar_books = book_similarities.nlargest(n)
        
        # Get book details for recommendations
        recommended_books = []
        for isbn in similar_books.index:
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                recommended_books.append(book_info.iloc[0])
        
        return pd.DataFrame(recommended_books)
    
    def hybrid_recommendations(self, user_id, n=10, user_weight=0.5, item_weight=0.5):
        """
        Generate hybrid recommendations using both user-based and item-based CF.
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n : int
            Number of recommendations to return
        user_weight : float
            Weight to give to user-based recommendations (0-1)
        item_weight : float
            Weight to give to item-based recommendations (0-1)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        # Get recommendations from both methods
        user_recs = self.user_based_recommendations(user_id, n=n*2)
        item_recs = self.item_based_recommendations(user_id, n=n*2)
        
        if user_recs.empty and item_recs.empty:
            return pd.DataFrame()
        elif user_recs.empty:
            return item_recs.head(n)
        elif item_recs.empty:
            return user_recs.head(n)
        
        # Combine the recommendations
        user_recs['method'] = 'user'
        item_recs['method'] = 'item'
        
        # Concatenate the dataframes
        all_recs = pd.concat([user_recs, item_recs])
        
        # Remove duplicates, keeping the one with the highest weight
        def get_weight(row):
            if row['method'] == 'user':
                return user_weight
            else:
                return item_weight
        
        all_recs['weight'] = all_recs.apply(get_weight, axis=1)
        
        # Sort by weight and remove duplicates
        all_recs = all_recs.sort_values('weight', ascending=False)
        all_recs = all_recs.drop_duplicates(subset=['ISBN'])
        
        # Return top n recommendations
        return all_recs.head(n).drop(['method', 'weight'], axis=1)


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
    preprocessor.process_all(min_book_ratings=5, min_user_ratings=5)
    
    # Initialize and fit the collaborative filtering recommender
    recommender = CollaborativeFilteringRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    recommender.fit()
    
    # Get recommendations for a user
    test_user = preprocessor.ratings_processed['User-ID'].iloc[0]
    recommendations = recommender.hybrid_recommendations(test_user, n=5)
    
    print(f"Recommendations for user {test_user}:")
    print(recommendations[['Book-Title', 'Book-Author']])
