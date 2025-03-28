import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class RecommendationsDisplay:
    """
    Comprehensive class to prepare and save different types of book recommendations for display.
    Handles popularity-based, content-based, and collaborative filtering recommendations.
    """
    
    def __init__(self, popularity_recommender=None, content_recommender=None, collaborative_recommender=None):
        """
        Initialize with trained recommender instances.
        
        Parameters:
        -----------
        popularity_recommender : PopularityRecommender, optional
            A trained popularity recommender instance
        content_recommender : ContentBasedRecommender, optional
            A trained content-based recommender instance
        collaborative_recommender : CollaborativeFilteringRecommender, optional
            A trained collaborative filtering recommender instance
        """
        self.popularity_recommender = popularity_recommender
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _standardize_book_format(self, books_df):
        """
        Standardize the format of book data for consistent display.
        
        Parameters:
        -----------
        books_df : pandas.DataFrame
            DataFrame containing book information
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with standardized column names
        """
        # Select relevant columns if they exist
        columns_to_select = []
        for col in ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                   'Publisher', 'Image-URL-M', 'rating_mean', 'rating_count', 'popularity_score']:
            if col in books_df.columns:
                columns_to_select.append(col)
        
        # Create a copy with only the selected columns
        display_books = books_df[columns_to_select].copy()
        
        # Create a mapping of old column names to new standardized names
        column_mapping = {
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year',
            'Image-URL-M': 'image_url',
            'rating_mean': 'average_rating',
            'rating_count': 'num_ratings'
        }
        
        # Rename only the columns that exist in the DataFrame
        for old_col, new_col in column_mapping.items():
            if old_col in display_books.columns:
                display_books.rename(columns={old_col: new_col}, inplace=True)
        
        return display_books
    
    #
    # POPULARITY-BASED RECOMMENDATIONS
    #
    
    def get_top_popular_books(self, n=100):
        """
        Get the top n books based on popularity score.
        
        Parameters:
        -----------
        n : int
            Number of books to return
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the top n books
        """
        if self.popularity_recommender is None:
            raise ValueError("Popularity recommender is not initialized")
            
        # Get recommendations from the popularity recommender
        top_books = self.popularity_recommender.recommend(n=n, criteria='popularity_score')
        
        # Standardize format
        return self._standardize_book_format(top_books)
    
    def get_popular_books_by_decade(self, n_per_decade=10, start_decade=1930, end_decade=2020):
        """
        Get the top n books for each decade from start_decade to end_decade.
        
        Parameters:
        -----------
        n_per_decade : int
            Number of books to return per decade
        start_decade : int
            Starting decade (e.g., 1930 for the 1930s)
        end_decade : int
            Ending decade (e.g., 2020 for the 2020s)
        
        Returns:
        --------
        dict
            Dictionary with decades as keys and DataFrames of top books as values
        """
        if self.popularity_recommender is None:
            raise ValueError("Popularity recommender is not initialized")
            
        decades = {}
        current_year = datetime.now().year
        
        # Ensure end_decade doesn't exceed the current decade
        end_decade = min(end_decade, (current_year // 10) * 10)
        
        for decade in range(start_decade, end_decade + 1, 10):
            # Get books published in this decade
            decade_books = self.popularity_recommender.popularity_df[
                (self.popularity_recommender.popularity_df['Year-Of-Publication'] >= decade) & 
                (self.popularity_recommender.popularity_df['Year-Of-Publication'] < decade + 10)
            ]
            
            if not decade_books.empty:
                # Sort by popularity score
                decade_books = decade_books.sort_values('popularity_score', ascending=False)
                
                # Select top n books
                top_decade_books = decade_books.head(n_per_decade)
                
                # Standardize format
                decades[str(decade)] = self._standardize_book_format(top_decade_books)
        
        return decades
    
    def save_popular_books_to_json(self, n=100, filename='top_popular_books.json'):
        """
        Save the top n popular books to a JSON file.
        
        Parameters:
        -----------
        n : int
            Number of books to save
        filename : str
            Name of the output JSON file
        """
        top_books = self.get_top_popular_books(n)
        
        # Convert to list of dictionaries for JSON serialization
        books_list = top_books.to_dict(orient='records')
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(books_list, f, ensure_ascii=False, indent=2)
        
        print(f"Saved top {n} popular books to {output_path}")
    
    def save_decade_books_to_json(self, n_per_decade=10, filename='decade_popular_books.json'):
        """
        Save the top n popular books for each decade to a JSON file.
        
        Parameters:
        -----------
        n_per_decade : int
            Number of books to save per decade
        filename : str
            Name of the output JSON file
        """
        decade_books = self.get_popular_books_by_decade(n_per_decade)
        
        # Convert DataFrames to lists of dictionaries for JSON serialization
        decade_dict = {}
        for decade, books_df in decade_books.items():
            decade_dict[decade] = books_df.to_dict(orient='records')
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(decade_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Saved top {n_per_decade} popular books per decade to {output_path}")
    
    #
    # CONTENT-BASED RECOMMENDATIONS
    #
    
    def get_similar_books_by_title(self, title, n=5):
        """
        Get books similar to the given title using content-based filtering.
        
        Parameters:
        -----------
        title : str
            Title of the book to find recommendations for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        tuple
            (source_book, similar_books) where source_book is the book used for recommendations
            and similar_books is a DataFrame of recommended books
        """
        if self.content_recommender is None:
            raise ValueError("Content-based recommender is not initialized")
            
        # Find books with matching titles (case insensitive)
        matching_books = self.content_recommender.books_df[
            self.content_recommender.books_df['Book-Title'].str.lower() == title.lower()
        ]
        
        if matching_books.empty:
            # Try partial matching if exact match fails
            matching_books = self.content_recommender.books_df[
                self.content_recommender.books_df['Book-Title'].str.lower().str.contains(title.lower())
            ]
            
            if matching_books.empty:
                print(f"No books found with title containing '{title}'.")
                return None, pd.DataFrame()
        
        # Use the first matching book
        source_book = matching_books.iloc[0]
        book_isbn = source_book['ISBN']
        
        # Get recommendations based on the ISBN
        similar_books = self.content_recommender.get_recommendations(book_isbn, n+1)
        
        # Remove the source book from recommendations if it's present
        similar_books = similar_books[similar_books['ISBN'] != book_isbn].head(n)
        
        # Standardize format
        return self._standardize_book_format(pd.DataFrame([source_book])).iloc[0], self._standardize_book_format(similar_books)
    
    def save_similar_books_to_json(self, titles, n=5, filename='similar_books.json'):
        """
        Save similar books for a list of titles to a JSON file.
        
        Parameters:
        -----------
        titles : list
            List of book titles to find similar books for
        n : int
            Number of similar books to save per title
        filename : str
            Name of the output JSON file
        """
        if self.content_recommender is None:
            raise ValueError("Content-based recommender is not initialized")
            
        similar_books_dict = {}
        
        for title in titles:
            source_book, similar_books = self.get_similar_books_by_title(title, n)
            
            if source_book is not None and not similar_books.empty:
                # Convert to dictionaries for JSON serialization
                source_book_dict = source_book.to_dict()
                similar_books_list = similar_books.to_dict(orient='records')
                
                # Store in the dictionary
                similar_books_dict[title] = {
                    'source_book': source_book_dict,
                    'similar_books': similar_books_list
                }
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(similar_books_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Saved similar books for {len(similar_books_dict)} titles to {output_path}")
    
    def create_similar_books_lookup(self, n=5, min_ratings=20, filename='similar_books_lookup.json'):
        """
        Create a lookup of similar books for popular books.
        
        Parameters:
        -----------
        n : int
            Number of similar books to save per book
        min_ratings : int
            Minimum number of ratings a book must have to be included
        filename : str
            Name of the output JSON file
        """
        if self.content_recommender is None or self.popularity_recommender is None:
            raise ValueError("Both content-based and popularity recommenders must be initialized")
            
        # Get popular books with minimum ratings
        popular_books = self.popularity_recommender.popularity_df[
            self.popularity_recommender.popularity_df['rating_count'] >= min_ratings
        ].sort_values('popularity_score', ascending=False).head(100)
        
        similar_books_dict = {}
        
        for _, book in popular_books.iterrows():
            isbn = book['ISBN']
            title = book['Book-Title']
            
            # Get similar books (get n+1 and then remove the source book)
            similar_books = self.content_recommender.get_recommendations(isbn, n+1)
            similar_books = similar_books[similar_books['ISBN'] != isbn].head(n)
            
            if not similar_books.empty:
                # Standardize format
                source_book_dict = self._standardize_book_format(pd.DataFrame([book])).iloc[0].to_dict()
                similar_books_list = self._standardize_book_format(similar_books).to_dict(orient='records')
                
                # Store in the dictionary
                similar_books_dict[isbn] = {
                    'source_book': source_book_dict,
                    'similar_books': similar_books_list
                }
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(similar_books_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Created similar books lookup for {len(similar_books_dict)} popular books in {output_path}")
    
    #
    # COLLABORATIVE FILTERING RECOMMENDATIONS
    #
    
    def get_user_based_recommendations(self, user_id, n=10, k=20):
        """
        Get user-based collaborative filtering recommendations for a user.
        
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
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative filtering recommender is not initialized")
            
        # Get recommendations from the collaborative recommender
        recommendations = self.collaborative_recommender.user_based_recommendations(user_id, n, k)
        
        if recommendations.empty:
            print(f"No user-based recommendations found for user {user_id}")
            return pd.DataFrame()
            
        # Standardize format
        return self._standardize_book_format(recommendations)
    
    def get_item_based_recommendations(self, user_id, n=10):
        """
        Get item-based collaborative filtering recommendations for a user.
        
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
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative filtering recommender is not initialized")
            
        # Get recommendations from the collaborative recommender
        recommendations = self.collaborative_recommender.item_based_recommendations(user_id, n)
        
        if recommendations.empty:
            print(f"No item-based recommendations found for user {user_id}")
            return pd.DataFrame()
            
        # Standardize format
        return self._standardize_book_format(recommendations)
    
    def get_hybrid_recommendations(self, user_id, n=10, user_weight=0.5):
        """
        Get hybrid collaborative filtering recommendations for a user.
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n : int
            Number of recommendations to return
        user_weight : float
            Weight to give to user-based recommendations (0-1)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recommended books
        """
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative filtering recommender is not initialized")
            
        # Get recommendations from the collaborative recommender
        recommendations = self.collaborative_recommender.hybrid_recommendations(user_id, n, user_weight)
        
        if recommendations.empty:
            print(f"No hybrid recommendations found for user {user_id}")
            return pd.DataFrame()
            
        # Standardize format
        return self._standardize_book_format(recommendations)
    
    def save_curated_recommendations_to_json(self, n=100, filename='curated_for_you.json'):
        """
        Save the top books based on collaborative filtering to a JSON file as "Curated Just for You".
        This uses the item similarity matrix to find the most universally appealing books.
        
        Parameters:
        -----------
        n : int
            Number of curated books to save
        filename : str
            Name of the output JSON file
        """
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative filtering recommender is not initialized")
            
        # Get the item similarity matrix
        item_similarity = self.collaborative_recommender.item_similarity
        
        if item_similarity is None or item_similarity.empty:
            raise ValueError("Item similarity matrix is not available. Make sure to fit the collaborative filtering recommender first.")
        
        print(f"Generating top {n} curated books based on collaborative filtering...")
        
        # Calculate the average similarity score for each book
        # This represents how similar a book is to all other books on average
        avg_similarity = item_similarity.mean(axis=1)
        
        # Sort books by their average similarity score
        top_similar_isbns = avg_similarity.sort_values(ascending=False).head(n*2).index
        
        # Get book details for these ISBNs
        curated_books = []
        for isbn in top_similar_isbns:
            book_info = self.collaborative_recommender.books_df[self.collaborative_recommender.books_df['ISBN'] == isbn]
            if not book_info.empty:
                curated_books.append(book_info.iloc[0])
                
                # Break if we have enough books
                if len(curated_books) >= n:
                    break
        
        # Convert to DataFrame
        curated_df = pd.DataFrame(curated_books)
        
        # Standardize format
        curated_df = self._standardize_book_format(curated_df)
        
        # Convert to list of dictionaries for JSON serialization
        curated_list = curated_df.to_dict(orient='records')
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(curated_list, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(curated_list)} curated books to {output_path}")
        return curated_df
    
    def save_collaborative_recommendations_to_json(self, user_ids, n=10, filename='collaborative_recommendations.json'):
        """
        Save collaborative filtering recommendations for a list of users to a JSON file.
        
        Parameters:
        -----------
        user_ids : list
            List of user IDs to get recommendations for
        n : int
            Number of recommendations to return per user
        filename : str
            Name of the output JSON file
        """
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative filtering recommender is not initialized")
            
        recommendations_dict = {}
        
        for user_id in user_ids:
            user_recommendations = {}
            
            # Get user-based recommendations
            user_based = self.get_user_based_recommendations(user_id, n)
            if not user_based.empty:
                user_recommendations['user_based'] = user_based.to_dict(orient='records')
                
            # Get item-based recommendations
            item_based = self.get_item_based_recommendations(user_id, n)
            if not item_based.empty:
                user_recommendations['item_based'] = item_based.to_dict(orient='records')
                
            # Get hybrid recommendations
            hybrid = self.get_hybrid_recommendations(user_id, n)
            if not hybrid.empty:
                user_recommendations['hybrid'] = hybrid.to_dict(orient='records')
                
            if user_recommendations:
                recommendations_dict[str(user_id)] = user_recommendations
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Saved collaborative filtering recommendations for {len(recommendations_dict)} users to {output_path}")
    
    #
    # SAVE ALL RECOMMENDATIONS
    #
    
    def save_all_recommendations(self, sample_user_ids=None, sample_titles=None):
        """
        Save all types of recommendations.
        
        Parameters:
        -----------
        sample_user_ids : list, optional
            List of user IDs to save collaborative filtering recommendations for
        sample_titles : list, optional
            List of book titles to save content-based recommendations for
        """
        # Save popularity-based recommendations
        if self.popularity_recommender is not None:
            self.save_popular_books_to_json(n=100)
            self.save_decade_books_to_json(n_per_decade=10)
            print("Popularity-based recommendations saved.")
        
        # Save content-based recommendations
        if self.content_recommender is not None:
            # Create a lookup of similar books for popular books
            self.create_similar_books_lookup(n=5, min_ratings=20)
            
            # Save similar books for sample titles if provided
            if sample_titles:
                self.save_similar_books_to_json(sample_titles, n=5)
            
            print("Content-based recommendations saved.")
        
        # Save collaborative filtering recommendations
        if self.collaborative_recommender is not None:
            # Save curated recommendations
            self.save_curated_recommendations_to_json(n=100)
            
            # Save user-specific recommendations if user IDs are provided
            if sample_user_ids:
                self.save_collaborative_recommendations_to_json(sample_user_ids, n=10)
            
            print("Collaborative filtering recommendations saved.")
        
        print("All recommendation data saved successfully.")


# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from popularity_based import PopularityRecommender
    from content_based import ContentBasedRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(
        books_path="../Books.csv",
        ratings_path="../Ratings.csv",
        users_path="../Users.csv"
    )
    preprocessor.process_all(min_book_ratings=10, min_user_ratings=5)
    
    # Initialize and fit the popularity recommender
    popularity_recommender = PopularityRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    popularity_recommender.fit(min_ratings=20)
    
    # Initialize and fit the content-based recommender
    content_recommender = ContentBasedRecommender(preprocessor.books_processed)
    content_recommender.fit()
    
    # Initialize and fit the collaborative filtering recommender
    collaborative_recommender = CollaborativeFilteringRecommender(preprocessor.ratings_processed, preprocessor.books_processed)
    collaborative_recommender.fit()
    
    # Create recommendations display and save all recommendations
    recommendations_display = RecommendationsDisplay(
        popularity_recommender=popularity_recommender,
        content_recommender=content_recommender,
        collaborative_recommender=collaborative_recommender
    )
    
    # Save all recommendations
    recommendations_display.save_all_recommendations(sample_user_ids=[1, 2, 3], sample_titles=["Harry Potter and the Sorcerer's Stone", "The Lord of the Rings"])
    
    # Example of getting similar books for a specific title
    title = "Harry Potter and the Sorcerer's Stone"
    source_book, similar_books = recommendations_display.get_similar_books_by_title(title, n=5)
    
    if source_book is not None:
        print(f"\nSimilar books to '{title}':")
        for _, book in similar_books.iterrows():
            print(f"- {book['title']} by {book['author']}")
    
    print("\nDone! Check the 'output' directory for the JSON files.")
