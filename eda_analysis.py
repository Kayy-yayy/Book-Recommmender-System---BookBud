import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from src.data_preprocessing import DataPreprocessor
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.popularity_based import PopularityRecommender

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create output directory for plots
output_dir = os.path.join(os.path.dirname(__file__), 'eda_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_plot(filename):
    """Save the current plot to the output directory."""
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_original_data(preprocessor):
    """Analyze the original dataset before filtering."""
    print("\n=== ORIGINAL DATASET ANALYSIS ===")
    
    # Basic statistics
    print(f"Total ratings: {len(preprocessor.ratings_processed):,}")
    print(f"Unique users: {preprocessor.ratings_processed['User-ID'].nunique():,}")
    print(f"Unique books: {preprocessor.ratings_processed['ISBN'].nunique():,}")
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    rating_counts = preprocessor.ratings_processed['Book-Rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('Distribution of Ratings in Original Dataset', fontsize=14)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    save_plot('original_rating_distribution.png')
    
    # User activity distribution
    plt.figure(figsize=(12, 6))
    user_activity = preprocessor.ratings_processed['User-ID'].value_counts()
    sns.histplot(user_activity, bins=50, log_scale=(False, True))
    plt.title('Distribution of Ratings per User (log scale)', fontsize=14)
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Number of Users (log scale)', fontsize=12)
    plt.axvline(x=200, color='red', linestyle='--', label='Min User Ratings Threshold (200)')
    plt.legend()
    save_plot('user_activity_distribution.png')
    
    # Book popularity distribution
    plt.figure(figsize=(12, 6))
    book_popularity = preprocessor.ratings_processed['ISBN'].value_counts()
    sns.histplot(book_popularity, bins=50, log_scale=(False, True))
    plt.title('Distribution of Ratings per Book (log scale)', fontsize=14)
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Number of Books (log scale)', fontsize=12)
    plt.axvline(x=50, color='red', linestyle='--', label='Min Book Ratings Threshold (50)')
    plt.legend()
    save_plot('book_popularity_distribution.png')
    
    # Publication year distribution
    if 'Year-Of-Publication' in preprocessor.books_processed.columns:
        plt.figure(figsize=(14, 6))
        # Convert to numeric, coerce errors to NaN
        years = pd.to_numeric(preprocessor.books_processed['Year-Of-Publication'], errors='coerce')
        # Filter out unreasonable years (e.g., future years or very old books)
        current_year = datetime.now().year
        valid_years = years[(years > 1800) & (years <= current_year)]
        sns.histplot(valid_years, bins=50)
        plt.title('Distribution of Book Publication Years', fontsize=14)
        plt.xlabel('Year of Publication', fontsize=12)
        plt.ylabel('Number of Books', fontsize=12)
        save_plot('publication_year_distribution.png')
    
    # Top publishers
    if 'Publisher' in preprocessor.books_processed.columns:
        plt.figure(figsize=(14, 8))
        top_publishers = preprocessor.books_processed['Publisher'].value_counts().head(20)
        sns.barplot(y=top_publishers.index, x=top_publishers.values)
        plt.title('Top 20 Publishers by Number of Books', fontsize=14)
        plt.xlabel('Number of Books', fontsize=12)
        plt.ylabel('Publisher', fontsize=12)
        save_plot('top_publishers.png')
    
    return user_activity, book_popularity

def analyze_filtered_data(cf_recommender):
    """Analyze the filtered dataset used for collaborative filtering."""
    print("\n=== FILTERED DATASET ANALYSIS ===")
    
    # Check if matrices are created
    if cf_recommender.item_user_matrix is None:
        print("Error: Item-user matrix not created. Run cf_recommender.create_matrices() first.")
        return
    
    # Matrix dimensions
    item_user_shape = cf_recommender.item_user_matrix.shape
    print(f"Item-user matrix dimensions: {item_user_shape} (Books × Users)")
    
    # Calculate sparsity
    total_cells = cf_recommender.item_user_matrix.size
    filled_cells = (cf_recommender.item_user_matrix > 0).sum().sum()
    sparsity = 100 * (1 - filled_cells / total_cells)
    print(f"Matrix sparsity: {sparsity:.2f}% ({filled_cells:,} non-zero entries out of {total_cells:,} total)")
    
    # Average ratings per user and per book in filtered data
    avg_ratings_per_user = (cf_recommender.item_user_matrix > 0).sum(axis=0).mean()
    avg_ratings_per_book = (cf_recommender.item_user_matrix > 0).sum(axis=1).mean()
    print(f"Average ratings per user in filtered data: {avg_ratings_per_user:.2f}")
    print(f"Average ratings per book in filtered data: {avg_ratings_per_book:.2f}")
    
    # Rating distribution in filtered data
    plt.figure(figsize=(10, 6))
    # Flatten the matrix to get all ratings
    all_ratings = cf_recommender.item_user_matrix.values.flatten()
    # Filter out zeros (no ratings)
    all_ratings = all_ratings[all_ratings > 0]
    sns.histplot(all_ratings, bins=10, kde=True)
    plt.title('Distribution of Ratings in Filtered Dataset', fontsize=14)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    save_plot('filtered_rating_distribution.png')
    
    return sparsity, avg_ratings_per_user, avg_ratings_per_book

def analyze_similarity_matrices(cf_recommender, content_recommender):
    """Analyze the similarity matrices from collaborative and content-based filtering."""
    print("\n=== SIMILARITY MATRICES ANALYSIS ===")
    
    # Analyze item similarity from collaborative filtering
    if hasattr(cf_recommender, 'item_similarity') and cf_recommender.item_similarity is not None:
        print("Analyzing collaborative filtering item similarity matrix...")
        
        # Get average similarity for each book
        avg_similarity = cf_recommender.item_similarity.mean(axis=1)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(avg_similarity, bins=50, kde=True)
        plt.title('Distribution of Average Book Similarity (Collaborative Filtering)', fontsize=14)
        plt.xlabel('Average Similarity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        save_plot('cf_avg_similarity_distribution.png')
        
        # Distribution of similarity values
        plt.figure(figsize=(12, 6))
        # Get upper triangle of similarity matrix to avoid duplicates
        similarity_values = cf_recommender.item_similarity.values
        # Only take a sample if the matrix is large
        if similarity_values.size > 1000000:  # If more than 1 million values
            # Take a random sample of 100,000 values
            flat_values = similarity_values.flatten()
            similarity_values = np.random.choice(flat_values, size=100000, replace=False)
        else:
            mask = np.triu_indices_from(similarity_values, k=1)
            similarity_values = similarity_values[mask]
        
        sns.histplot(similarity_values, bins=50, kde=True)
        plt.title('Distribution of Book-Book Similarity Values (Collaborative Filtering)', fontsize=14)
        plt.xlabel('Similarity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        save_plot('cf_similarity_distribution.png')
        
        # Print top books by average similarity
        print("\nTop 10 books with highest average similarity (Collaborative Filtering):")
        top_books_cf = avg_similarity.sort_values(ascending=False).head(10)
        for isbn, sim_score in top_books_cf.items():
            book_info = cf_recommender.books_df[cf_recommender.books_df['ISBN'] == isbn]
            if not book_info.empty:
                print(f"- {book_info['Book-Title'].values[0]} by {book_info['Book-Author'].values[0]} (Avg Similarity: {sim_score:.4f})")
    
    # Analyze item similarity from content-based filtering
    if content_recommender is not None and hasattr(content_recommender, 'cosine_sim') and content_recommender.cosine_sim is not None:
        print("\nAnalyzing content-based filtering similarity matrix...")
        
        # Get average similarity for each book
        if isinstance(content_recommender.cosine_sim, pd.DataFrame):
            avg_content_similarity = content_recommender.cosine_sim.mean(axis=1)
        else:
            # If it's a numpy array
            avg_content_similarity = np.mean(content_recommender.cosine_sim, axis=1)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(avg_content_similarity, bins=50, kde=True)
        plt.title('Distribution of Average Book Similarity (Content-Based)', fontsize=14)
        plt.xlabel('Average Similarity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        save_plot('content_avg_similarity_distribution.png')
        
        # Distribution of similarity values
        plt.figure(figsize=(12, 6))
        # Get upper triangle of similarity matrix to avoid duplicates
        if isinstance(content_recommender.cosine_sim, pd.DataFrame):
            similarity_values = content_recommender.cosine_sim.values
        else:
            similarity_values = content_recommender.cosine_sim
            
        # Only take a sample if the matrix is large
        if similarity_values.size > 1000000:  # If more than 1 million values
            # Take a random sample of 100,000 values
            flat_values = similarity_values.flatten()
            similarity_values = np.random.choice(flat_values, size=100000, replace=False)
        else:
            mask = np.triu_indices_from(similarity_values, k=1)
            similarity_values = similarity_values[mask]
        
        sns.histplot(similarity_values, bins=50, kde=True)
        plt.title('Distribution of Book-Book Similarity Values (Content-Based)', fontsize=14)
        plt.xlabel('Similarity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        save_plot('content_similarity_distribution.png')
        
        # Print top books by average similarity
        print("\nTop 10 books with highest average similarity (Content-Based):")
        if isinstance(content_recommender.cosine_sim, pd.DataFrame):
            indices = content_recommender.indices
            top_books_content = avg_content_similarity.sort_values(ascending=False).head(10)
            for isbn, sim_score in top_books_content.items():
                book_info = content_recommender.books_df[content_recommender.books_df['ISBN'] == isbn]
                if not book_info.empty:
                    print(f"- {book_info['Book-Title'].values[0]} by {book_info['Book-Author'].values[0]} (Avg Similarity: {sim_score:.4f})")

def analyze_recommendations(popularity_recommender, content_recommender, cf_recommender, books_df):
    """Analyze and compare recommendations from different algorithms."""
    print("\n=== RECOMMENDATION ANALYSIS ===")
    
    # Get top popular books
    top_popular = popularity_recommender.recommend(n=20)
    print(f"\nTop 5 popular books:")
    for i, (_, book) in enumerate(top_popular.head(5).iterrows()):
        print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']} (Score: {book['popularity_score']:.4f})")
    
    # Analyze "Curated Just for You" recommendations if we have item similarity
    if hasattr(cf_recommender, 'item_similarity') and cf_recommender.item_similarity is not None:
        print("\nAnalyzing 'Curated Just for You' recommendations...")
        # Calculate average similarity for each book
        avg_similarity = cf_recommender.item_similarity.mean(axis=1)
        # Sort books by average similarity
        top_similar_isbns = avg_similarity.sort_values(ascending=False).head(20).index
        
        print("Top 5 'Curated Just for You' books (highest average similarity):")
        for i, isbn in enumerate(top_similar_isbns[:5]):
            book_info = cf_recommender.books_df[cf_recommender.books_df['ISBN'] == isbn]
            if not book_info.empty:
                book = book_info.iloc[0]
                print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
    
    # Get a random book for content-based recommendations
    if content_recommender is not None and len(books_df) > 0:
        # Try to find a popular book that's also in our content-based sample
        if not top_popular.empty:
            for _, book in top_popular.iterrows():
                random_isbn = book['ISBN']
                random_title = book['Book-Title']
                # Check if this book is in our content-based recommender
                if random_isbn in content_recommender.books_df['ISBN'].values:
                    print(f"\nSelected book for content-based recommendations: {random_title}")
                    break
            else:
                # If no match found, just take a random book from content recommender
                random_book = content_recommender.books_df.sample(1).iloc[0]
                random_isbn = random_book['ISBN']
                random_title = random_book['Book-Title']
                print(f"\nSelected random book for content-based recommendations: {random_title}")
        else:
            random_book = content_recommender.books_df.sample(1).iloc[0]
            random_isbn = random_book['ISBN']
            random_title = random_book['Book-Title']
            print(f"\nSelected random book for content-based recommendations: {random_title}")
        
        # Get content-based recommendations
        try:
            content_recs = content_recommender.get_recommendations(random_isbn, n=5)
            print("Top 5 similar books (content-based):")
            for i, (_, book) in enumerate(content_recs.iterrows()):
                print(f"{i+1}. {book['Book-Title']} by {book['Book-Author']}")
        except Exception as e:
            print(f"Error getting content-based recommendations: {e}")
            content_recs = pd.DataFrame()
    else:
        content_recs = pd.DataFrame()
    
    # Compare recommendation overlap
    if not content_recs.empty and not top_popular.empty:
        popular_isbns = set(top_popular['ISBN'])
        content_isbns = set(content_recs['ISBN'])
        overlap = popular_isbns.intersection(content_isbns)
        
        print(f"\nOverlap between popularity and content-based recommendations: {len(overlap)} books")
        print(f"Percentage overlap: {100 * len(overlap) / len(content_recs):.2f}%")
        
        # Create Venn diagram
        try:
            from matplotlib_venn import venn2
            plt.figure(figsize=(10, 6))
            venn2([popular_isbns, content_isbns], ('Popularity-based', 'Content-based'))
            plt.title('Overlap Between Recommendation Algorithms', fontsize=14)
            save_plot('recommendation_overlap.png')
        except ImportError:
            print("matplotlib_venn not installed. Skipping Venn diagram.")

def main():
    """Main EDA function."""
    print("Starting Exploratory Data Analysis for BookBud Recommendation System")
    print(f"Plots will be saved to: {output_dir}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = DataPreprocessor(
        books_path="Books.csv",
        ratings_path="Ratings.csv",
        users_path="Users.csv"
    )
    preprocessor.process_all()
    
    # Analyze original data
    user_activity, book_popularity = analyze_original_data(preprocessor)
    
    # Initialize collaborative filtering with new thresholds
    print("\nInitializing collaborative filtering recommender...")
    cf_recommender = CollaborativeFilteringRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    # Use the less strict filtering criteria
    cf_recommender.create_matrices(min_user_ratings=20, min_book_ratings=10)
    
    # Analyze filtered data
    sparsity, avg_user_ratings, avg_book_ratings = analyze_filtered_data(cf_recommender)
    
    # Initialize content-based recommender with a sample of books to avoid memory issues
    print("\nInitializing content-based recommender with a sample of books...")
    # Take a sample of 2000 books to avoid memory issues, but ensure we have enough for analysis
    sample_size = min(2000, len(preprocessor.books_processed))
    sample_books = preprocessor.books_processed.sample(sample_size).reset_index(drop=True)
    try:
        content_recommender = ContentBasedRecommender(sample_books)
        content_recommender.fit()
        print(f"Content-based recommender initialized with {sample_size} books")
    except Exception as e:
        print(f"Error fitting content-based recommender: {e}")
        content_recommender = None
    
    # Initialize popularity-based recommender
    print("\nInitializing popularity-based recommender...")
    popularity_recommender = PopularityRecommender(
        preprocessor.ratings_processed,
        preprocessor.books_processed
    )
    popularity_recommender.fit()
    
    # Analyze similarity matrices
    try:
        cf_recommender.compute_item_similarity()
        analyze_similarity_matrices(cf_recommender, content_recommender)
    except Exception as e:
        print(f"Error analyzing similarity matrices: {e}")
    
    # Analyze recommendations
    try:
        analyze_recommendations(popularity_recommender, content_recommender, cf_recommender, preprocessor.books_processed)
    except Exception as e:
        print(f"Error analyzing recommendations: {e}")
    
    print(f"\nEDA complete! All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
