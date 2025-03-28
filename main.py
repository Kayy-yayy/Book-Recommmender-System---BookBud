import os
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.popularity_based import PopularityRecommender
from src.recommendations_display import RecommendationsDisplay

def main():
    """
    Main function to run the BookBud recommendation system.
    """
    print("Starting BookBud Recommendation System...")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = DataPreprocessor(
        books_path="Books.csv",
        ratings_path="Ratings.csv",
        users_path="Users.csv"
    )
    preprocessor.process_all()
    
    print(f"Processed {len(preprocessor.books_processed)} books, "
          f"{len(preprocessor.ratings_processed)} ratings, and "
          f"{len(preprocessor.users_processed)} users.")
    
    # Initialize collaborative filtering with optimized parameters
    print("\nInitializing collaborative filtering recommender...")
    cf_recommender = CollaborativeFilteringRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    
    # Use the optimized filtering criteria
    cf_recommender.fit(min_user_ratings=20, min_book_ratings=10)
    
    # Initialize content-based recommender with a sample of books to avoid memory issues
    print("\nInitializing content-based recommender...")
    # Take a sample of books to avoid memory issues
    sample_size = min(2000, len(preprocessor.books_processed))
    sample_books = preprocessor.books_processed.sample(sample_size).reset_index(drop=True)
    content_recommender = ContentBasedRecommender(sample_books)
    content_recommender.fit()
    print(f"Content-based recommender initialized with {sample_size} books")
    
    # Initialize popularity-based recommender
    print("\nInitializing popularity-based recommender...")
    popularity_recommender = PopularityRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    popularity_recommender.fit()
    
    # Initialize recommendations display
    recommendations_display = RecommendationsDisplay(
        popularity_recommender=popularity_recommender,
        content_recommender=content_recommender,
        collaborative_recommender=cf_recommender
    )
    
    # Generate and save all recommendations
    print("\nGenerating and saving recommendations...")
    
    # Save popularity-based recommendations
    recommendations_display.save_popularity_recommendations_to_json()
    recommendations_display.save_decade_recommendations_to_json()
    
    # Save curated recommendations based on collaborative filtering
    recommendations_display.save_curated_recommendations_to_json()
    
    # Example: Get recommendations for a specific book
    example_book_title = "Harry Potter and the Sorcerer's Stone"
    print(f"\nGetting content-based recommendations for '{example_book_title}'...")
    try:
        similar_books = recommendations_display.get_similar_books_by_title(example_book_title, n=5)
        if not similar_books.empty:
            print(f"Top 5 books similar to '{example_book_title}':")
            for i, (_, book) in enumerate(similar_books.iterrows()):
                print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
        else:
            print(f"No similar books found for '{example_book_title}'")
    except Exception as e:
        print(f"Error getting similar books: {e}")
    
    # Example: Get collaborative filtering recommendations for a user
    if len(preprocessor.ratings_processed) > 0:
        example_user = preprocessor.ratings_processed['User-ID'].iloc[0]
        print(f"\nGetting collaborative filtering recommendations for user {example_user}...")
        
        # User-based recommendations
        try:
            user_based_recs = recommendations_display.get_user_based_recommendations(example_user, n=5)
            if not user_based_recs.empty:
                print(f"\nTop 5 user-based recommendations for user {example_user}:")
                for i, (_, book) in enumerate(user_based_recs.iterrows()):
                    print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
            else:
                print(f"No user-based recommendations found for user {example_user}")
        except Exception as e:
            print(f"Error getting user-based recommendations: {e}")
        
        # Item-based recommendations
        try:
            item_based_recs = recommendations_display.get_item_based_recommendations(example_user, n=5)
            if not item_based_recs.empty:
                print(f"\nTop 5 item-based recommendations for user {example_user}:")
                for i, (_, book) in enumerate(item_based_recs.iterrows()):
                    print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
            else:
                print(f"No item-based recommendations found for user {example_user}")
        except Exception as e:
            print(f"Error getting item-based recommendations: {e}")
        
        # Hybrid recommendations
        try:
            hybrid_recs = recommendations_display.get_hybrid_recommendations(example_user, n=5)
            if not hybrid_recs.empty:
                print(f"\nTop 5 hybrid recommendations for user {example_user}:")
                for i, (_, book) in enumerate(hybrid_recs.iterrows()):
                    print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
            else:
                print(f"No hybrid recommendations found for user {example_user}")
        except Exception as e:
            print(f"Error getting hybrid recommendations: {e}")
    
    print("\nAll recommendations have been generated and saved to the output directory.")
    print("BookBud Recommendation System completed successfully!")

if __name__ == "__main__":
    main()
