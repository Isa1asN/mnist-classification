import numpy as np
import math
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

class NaiveBayes:
    def __init__(self, train, train_lb, test, test_lb, smoothing,) -> None:
        self.n_class = np.unique(train_lb)
        self.train = train
        self.test = test
        self.train_lb = train_lb
        self.test_lb = test_lb
        self.smoothing = smoothing

    def cal_mean(self, data):
        num_samples = len(data)
        num_features = len(data[0])
        mean_values = [0.0] * num_features

        for sample in data:
            for i, feature in enumerate(sample):
                mean_values[i] += feature

        mean_values = [mean_val / num_samples for mean_val in mean_values]
        return mean_values


    def cal_std(self, data, mean_values):
        num_samples = len(data)
        num_features = len(data[0])
        squared_diff_sum = [0.0] * num_features

        for sample in data:
            for i, feature in enumerate(sample):
                squared_diff_sum[i] += (feature - mean_values[i]) ** 2

        std_deviation = [math.sqrt(squared_diff_sum_val / (num_samples - 1)) for squared_diff_sum_val in squared_diff_sum]
        return std_deviation


    def mean_std_prior(self):
        self.mean, self.std, self.priors, self.count = [], [], [], []
        for label in self.n_class:
            sep = [sample for sample, sample_label in zip(self.train, self.train_lb) if sample_label == label]
            count = len(sep)
            prior = count / len(self.train_lb)
            mean_values = self.cal_mean(sep)
            std_deviation = self.cal_std(sep, mean_values)

            self.mean.append(mean_values)
            self.std.append(std_deviation)
            self.priors.append(prior)
            self.count.append(count)

    def probability(self, x, mean, var):
        return 1 / np.sqrt(2 * math.pi * var) * np.exp(-np.square(x - mean)/(2 * var))

    def predict(self):
        print('Naive bayes is running...')
        self.mean_std_prior()
        self.pred = []
        self.likelihood = []
        self.logsum_chk = []
        for n in range(len(self.test_lb)):
            classifier = []
            sample = self.test[n] #test sample
            likelih = []
            for i, val in enumerate(self.n_class):
                mean = self.mean[i]
                var = np.square(self.std[i]) + self.smoothing
                prob = self.probability(sample, mean, var)
                result = np.sum(np.log(prob)) 
                classifier.append(result)
                likelih.append(prob)

            self.pred.append(np.argmax(classifier))
            self.likelihood.append(likelih)
            self.logsum_chk.append(classifier)



util = util.Util()
laplace_smoothing = 1000
feature_ext = FeatureExtraction(util.train, util.test).average_pixel_intensity()
naive_bayes = NaiveBayes(feature_ext[0], util.train_label, feature_ext[1], util.test_label, laplace_smoothing)
naive_bayes.predict()
accuracy = util.accuracy(util.test_label, naive_bayes.pred)

