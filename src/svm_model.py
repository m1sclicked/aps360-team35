import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

class GestureSVM:
    def __init__(self, num_classes=10, input_dim=126):
        """
        SVM model for gesture recognition, with an interface compatible with PyTorch models
        
        Args:
            num_classes: Number of gesture classes to recognize
            input_dim: Input feature dimension (not used but kept for compatibility with CNN interface)
        """
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.device = "cpu"  # SVMs always run on CPU
        
        # Initialize SVM pipeline with scaler
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, kernel='rbf', C=10, gamma='scale'))
        ])
        
        # For tracking training progress
        self.is_trained = False
        self.grid_search_results = None
        
        # Placeholders for metrics (to keep compatible interface with PyTorch models)
        self.train_losses = [0.0]  # SVMs don't have losses like NNs
        self.val_losses = [0.0]  # Placeholder 
        self.val_accuracies = []
        self.test_accuracies = []
        
    def to(self, device):
        """
        Mock method to maintain compatibility with PyTorch
        SVM always runs on CPU, but we'll keep the interface
        """
        self.device = device
        return self
        
    def train(self):
        """
        Set model to training mode (compatibility with PyTorch)
        """
        return self
        
    def eval(self):
        """
        Set model to evaluation mode (compatibility with PyTorch)
        """
        return self
        
    def parameters(self):
        """
        Mock method to maintain compatibility with PyTorch optimizers
        Returns an empty list as SVM doesn't use gradient-based optimization
        """
        return []
    
    def state_dict(self):
        """
        Return a serializable state dict for saving the model
        """
        return {"pipeline": self.pipeline}
        
    def load_state_dict(self, state_dict):
        """
        Load model from state dict
        """
        self.pipeline = state_dict["pipeline"]
        self.is_trained = True
        
    def forward(self, x):
        """
        Forward pass for compatibility with PyTorch
        In practice, this shouldn't be called directly
        """
        pass
    
    def fit(self, X, y, validation_data=None, test_data=None):
        """
        Train the SVM model
        """
        print("Training SVM model...")
        start_time = time.time()
        
        self.pipeline.fit(X, y)
        self.is_trained = True
        
        # Calculate validation accuracy if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_preds = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val) * 100
            self.val_accuracies.append(val_acc)
        
        # Calculate test accuracy if provided
        if test_data is not None:
            X_test, y_test = test_data
            test_preds = self.predict(X_test)
            test_acc = np.mean(test_preds == y_test) * 100
            self.test_accuracies.append(test_acc)
        
        training_time = time.time() - start_time
        print(f"SVM training completed in {training_time:.2f} seconds")
        return self
    
    def grid_search(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None, param_grid=None, cv=5):
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
        
        Returns:
            self: The trained model with best parameters
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import learning_curve
        
        print("Starting SVM GridSearchCV...")
        start_time = time.time()
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'svm__C': [0.1, 1, 10, 100, 1000],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svm__kernel': ['rbf']
            }
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            self.pipeline, 
            param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1  # Use all processors
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Get best model
        self.pipeline = grid_search.best_estimator_
        self.is_trained = True
        
        # Store grid search results
        self.grid_search_results = grid_search
        
        # Generate learning curves for best model
        print("Generating learning curves...")
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(
            self.pipeline, X, y, 
            train_sizes=train_sizes, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Store learning curve results
        self.grid_search_results.learning_curve_results = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        }
        
        # Compute validation and test accuracies
        self.val_accuracies = [grid_search.best_score_ * 100]
        
        if X_test is not None and y_test is not None:
            test_preds = self.predict(X_test)
            test_acc = np.mean(test_preds == y_test) * 100
            self.test_accuracies = [test_acc]
        else:
            self.test_accuracies = [0.0]
        
        training_time = time.time() - start_time
        print(f"Grid search completed in {training_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_*100:.2f}%")
        if X_test is not None:
            print(f"Test accuracy: {self.test_accuracies[0]:.2f}%")
        
        return self
        
    def predict_proba(self, X):
        """
        Get probability predictions from SVM
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict_proba(X)
        
    def predict(self, X):
        """
        Get class predictions from SVM
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict(X)