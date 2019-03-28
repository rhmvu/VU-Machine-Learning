class Result():
    def __init__(self, score, mean_squared_error, median_absolute_error):
        self.score = score
        self.mean_squared_error = mean_squared_error
        self.medium_absolute_error = median_absolute_error

    def mse(self):
        return self.mean_squared_error

    def score(self):
        return self.score

    def mae(self):
        return self.medium_absolute_error
