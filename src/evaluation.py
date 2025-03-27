import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderEvaluator:
    """
    Evaluator for recommendation systems.
    Provides methods to evaluate and compare different recommendation algorithms.
    """
    
    def __init__(self, ratings_df):
        """
        Initialize the evaluator with a ratings dataframe.
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings with columns: User-ID, ISBN, Book-Rating
        """
        self.ratings_df = ratings_df
        self.train_data = None
        self.test_data = None
        
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the ratings data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
        """
        # Split the data
        self.train_data, self.test_data = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Split data into {len(self.train_data)} training samples and {len(self.test_data)} testing samples.")
        return self.train_data, self.test_data
    
    def calculate_rmse(self, predictions, actual):
        """
        Calculate Root Mean Squared Error.
        
        Parameters:
        -----------
        predictions : array-like
            Predicted ratings
        actual : array-like
            Actual ratings
            
        Returns:
        --------
        float
            RMSE value
        """
        return np.sqrt(mean_squared_error(actual, predictions))
    
    def calculate_mae(self, predictions, actual):
        """
        Calculate Mean Absolute Error.
        
        Parameters:
        -----------
        predictions : array-like
            Predicted ratings
        actual : array-like
            Actual ratings
            
        Returns:
        --------
        float
            MAE value
        """
        return mean_absolute_error(actual, predictions)
    
    def evaluate_recommender(self, recommender_name, predict_func):
        """
        Evaluate a recommender system using RMSE and MAE.
        
        Parameters:
        -----------
        recommender_name : str
            Name of the recommender system
        predict_func : function
            Function that takes user_id and isbn and returns a predicted rating
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.test_data is None:
            print("Error: You must call train_test_split() before evaluate_recommender().")
            return {}
        
        # Get predictions for test data
        predictions = []
        actual = []
        
        for _, row in self.test_data.iterrows():
            user_id = row['User-ID']
            isbn = row['ISBN']
            true_rating = row['Book-Rating']
            
            try:
                predicted_rating = predict_func(user_id, isbn)
                predictions.append(predicted_rating)
                actual.append(true_rating)
            except Exception as e:
                # Skip this prediction if there's an error
                continue
        
        # Calculate metrics
        rmse = self.calculate_rmse(predictions, actual)
        mae = self.calculate_mae(predictions, actual)
        
        print(f"Evaluation results for {recommender_name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return {
            'recommender': recommender_name,
            'rmse': rmse,
            'mae': mae,
            'predictions': len(predictions)
        }
    
    def compare_recommenders(self, recommenders):
        """
        Compare multiple recommender systems.
        
        Parameters:
        -----------
        recommenders : list of tuples
            List of (recommender_name, predict_func) tuples
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing evaluation metrics for all recommenders
        """
        results = []
        
        for name, predict_func in recommenders:
            result = self.evaluate_recommender(name, predict_func)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results):
        """
        Plot comparison of recommender systems.
        
        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame containing evaluation metrics for all recommenders
        """
        plt.figure(figsize=(12, 6))
        
        # Plot RMSE
        plt.subplot(1, 2, 1)
        sns.barplot(x='recommender', y='rmse', data=results)
        plt.title('RMSE Comparison')
        plt.xlabel('Recommender')
        plt.ylabel('RMSE (lower is better)')
        plt.xticks(rotation=45)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        sns.barplot(x='recommender', y='mae', data=results)
        plt.title('MAE Comparison')
        plt.xlabel('Recommender')
        plt.ylabel('MAE (lower is better)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_top_n_recommendations(self, recommender_name, recommend_func, n=10):
        """
        Evaluate a recommender system based on top-N recommendations.
        
        Parameters:
        -----------
        recommender_name : str
            Name of the recommender system
        recommend_func : function
            Function that takes user_id and returns a list of recommended ISBNs
        n : int
            Number of recommendations to consider
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.test_data is None:
            print("Error: You must call train_test_split() before evaluate_top_n_recommendations().")
            return {}
        
        # Get unique users in test set
        test_users = self.test_data['User-ID'].unique()
        
        # Calculate metrics
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        user_count = 0
        
        for user_id in test_users:
            # Get books the user has rated in the test set
            user_test_books = set(self.test_data[self.test_data['User-ID'] == user_id]['ISBN'])
            
            # Skip users with no test books
            if len(user_test_books) == 0:
                continue
            
            # Get top-N recommendations for the user
            try:
                recommended_books = set(recommend_func(user_id, n))
                
                # Calculate precision and recall
                relevant_recommended = user_test_books.intersection(recommended_books)
                precision = len(relevant_recommended) / len(recommended_books) if recommended_books else 0
                recall = len(relevant_recommended) / len(user_test_books) if user_test_books else 0
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                user_count += 1
            except Exception as e:
                # Skip this user if there's an error
                continue
        
        # Calculate average metrics
        avg_precision = precision_sum / user_count if user_count > 0 else 0
        avg_recall = recall_sum / user_count if user_count > 0 else 0
        avg_f1 = f1_sum / user_count if user_count > 0 else 0
        
        print(f"Top-{n} evaluation results for {recommender_name}:")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  F1 Score: {avg_f1:.4f}")
        
        return {
            'recommender': recommender_name,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'users_evaluated': user_count
        }
    
    def compare_top_n_recommenders(self, recommenders, n=10):
        """
        Compare multiple recommender systems based on top-N recommendations.
        
        Parameters:
        -----------
        recommenders : list of tuples
            List of (recommender_name, recommend_func) tuples
        n : int
            Number of recommendations to consider
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing evaluation metrics for all recommenders
        """
        results = []
        
        for name, recommend_func in recommenders:
            result = self.evaluate_top_n_recommendations(name, recommend_func, n)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_top_n_comparison(self, results):
        """
        Plot comparison of recommender systems based on top-N recommendations.
        
        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame containing evaluation metrics for all recommenders
        """
        plt.figure(figsize=(15, 5))
        
        # Plot Precision
        plt.subplot(1, 3, 1)
        sns.barplot(x='recommender', y='precision', data=results)
        plt.title('Precision Comparison')
        plt.xlabel('Recommender')
        plt.ylabel('Precision (higher is better)')
        plt.xticks(rotation=45)
        
        # Plot Recall
        plt.subplot(1, 3, 2)
        sns.barplot(x='recommender', y='recall', data=results)
        plt.title('Recall Comparison')
        plt.xlabel('Recommender')
        plt.ylabel('Recall (higher is better)')
        plt.xticks(rotation=45)
        
        # Plot F1 Score
        plt.subplot(1, 3, 3)
        sns.barplot(x='recommender', y='f1', data=results)
        plt.title('F1 Score Comparison')
        plt.xlabel('Recommender')
        plt.ylabel('F1 Score (higher is better)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


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
    
    # Initialize the evaluator
    evaluator = RecommenderEvaluator(preprocessor.ratings_processed)
    
    # Split data into training and testing sets
    train_data, test_data = evaluator.train_test_split(test_size=0.2)
    
    # Example dummy recommender functions
    def random_predictor(user_id, isbn):
        return np.random.randint(1, 11)
    
    def mean_predictor(user_id, isbn):
        return preprocessor.ratings_processed['Book-Rating'].mean()
    
    # Evaluate recommenders
    results = evaluator.compare_recommenders([
        ('Random', random_predictor),
        ('Mean', mean_predictor)
    ])
    
    print("\nComparison Results:")
    print(results)
