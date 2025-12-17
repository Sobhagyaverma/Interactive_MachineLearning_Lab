import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        # We store the history of centroids to animate the movement!
        self.history = []

    def fit(self, X):
        """
        Run the K-Means algorithm:
        1. Initialize centroids randomly
        2. Loop until convergence (Assign -> Move -> Repeat)
        """
        n_samples, n_features = X.shape
        
        # 1. Randomly choose 'k' points from X to be the starting centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        self.history = [] # Reset history
        
        for i in range(self.max_iters):
            # Save current position for animation
            self.history.append(self.centroids.copy())
            
            # --- STEP A: ASSIGNMENT (Find closest centroid) ---
            # We calculate distance from every point to every centroid
            # labels will be a list of 0, 1, or 2 depending on which centroid is closest
            self.labels = self._assign_clusters(X, self.centroids)
            
            # --- STEP B: UPDATE (Move centroids) ---
            new_centroids = np.zeros((self.k, n_features))
            for cluster_idx in range(self.k):
                # Get all points belonging to this cluster
                cluster_points = X[self.labels == cluster_idx]
                
                # If a cluster has points, move centroid to the mean (average)
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    # Rare edge case: If a cluster is empty, keep old centroid
                    new_centroids[cluster_idx] = self.centroids[cluster_idx]
            
            # --- STEP C: CHECK CONVERGENCE ---
            # If centroids didn't move at all, we are done!
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster for new points (used for drawing the background)
        """
        return self._assign_clusters(X, self.centroids)

    def _assign_clusters(self, X, centroids):
        """
        Helper: Calculates distance from X to all centroids and returns the index of the closest one.
        """
        # Calculate distance between every point and every centroid
        distances = np.zeros((X.shape[0], self.k))
        
        for idx, centroid in enumerate(centroids):
            # Euclidean Distance: sqrt(sum((x - centroid)^2))
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)
            
        # Return index of minimum distance (0, 1, or 2)
        return np.argmin(distances, axis=1)