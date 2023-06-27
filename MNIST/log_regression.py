import numpy as np
import util


class FeatureExtraction:
    def __init__(self, train, test) -> None:
        self.train = train 
        self.test = test 
    
    def normalized_pixel_intensity(self):
        self.train = self.train / 255
        self.test = self.test / 255
        print('Features extracted')
        return self.train, self.test

    def nor_by_avg(self, features):
        avg_intensity = np.mean(features, axis=1)
        normalized_avg_intensity = avg_intensity / 255
        normalized_avg_intensity = np.reshape(normalized_avg_intensity, (-1, 1))
        normalized_features = np.hstack((features, normalized_avg_intensity))
        return normalized_features

    def average_pixel_intensity(self):
        self.train = self.nor_by_avg(self.train)
        self.test = self.nor_by_avg(self.test)
        print('Features extracted')
        return self.train, self.test
    
    def raw_pixel_values(self):
        return self.train, self.test


class LogisticRegression:
    def __init__(self, learning_rate, num_iterations) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        print('Logistic regression model training...')

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
        
    def calculate_distance(self, weights1, weights2):
        return np.linalg.norm(weights1 - weights2)
    
    def calculate_gradient(self, w, features):
        return np.dot(w, features)
    
    def multiply_arr(self, a, b):
        return np.multiply(a, b)
        
    def train(self, features, target, lambda_value):
        weights = np.reshape(np.ones(features.shape[1]), (1, features.shape[1]))
        iteration = 1
        distance = 1
        while distance > 0.0000001 and iteration < self.num_iterations:
            old_weights = weights.copy()
            scores = self.calculate_gradient(features, weights.T)
            weighted_scores = self.multiply_arr(target.T, scores)
            probabilities = self.sigmoid(weighted_scores)
            complement_probabilities = (1 - probabilities).T
            weighted_complement = self.multiply_arr(target, complement_probabilities)
            gradient = -self.calculate_gradient(weighted_complement, features)
            lambda_weight = lambda_value * weights
            gradient_lambda_weight = gradient + lambda_weight
            smoothed_gradient = self.learning_rate * gradient_lambda_weight
            weights = weights - smoothed_gradient
            distance = self.calculate_distance(old_weights, weights)
            iteration += 1

        return weights

    def predict(self, features, train_labels, test_features, test_labels, add_bias=True):
        lambda_value = 0.1
        if add_bias:  # add column of ones for bias term (w0)
            features = np.hstack((np.ones((features.shape[0], 1)), features))
            test_features = np.hstack((np.ones((test_features.shape[0], 1)), test_features))
        probabilities = np.zeros((test_features.shape[0], 10))
        for class_label in range(10):
            train_target = np.reshape(np.where(train_labels == class_label, 1, -1), (1, train_labels.shape[0]))
            weights_logistic = self.train(features, train_target, lambda_value)
            scores = np.dot(test_features, weights_logistic.T)
            pred_prob = self.sigmoid(scores)
            probabilities[:, class_label] = pred_prob[:, 0]  

        probability = np.asarray(probabilities)
        overall_pred_class = np.argmax(probability, axis=1)
        return overall_pred_class
    
util = util.Util()
feature_ext = FeatureExtraction(util.train, util.test).normalized_pixel_intensity()
logistic = LogisticRegression(1.5, 500)
result = logistic.predict(feature_ext[0], util.train_label, feature_ext[1], util.test_label)
accuracy = util.accuracy(util.test_label, result)
