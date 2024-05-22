
class PerceptronNot:
    def __init__(self, learning_rate, n_epoch, error_min):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.error_min = error_min
        self.weights = [0.01, 0] # bias, weight
    
    def activation_function(self, weighted_sum):
        # step function
        if weighted_sum > 0:
            ret = 1.0
        else:
            ret = 0 
        return ret
    
    def calculate_weighted_sum(self, inputs):
        weighted_sum = self.weights[0] + inputs[0] * self.weights[1]
        return weighted_sum
            
    def learn(self, x, y_expected):
        y_calculated = self.output(x)
        error = y_expected - y_calculated
        print(f"Input: {x}, y_expected: {y_expected}, error: {error}, weights: {self.weights}")
        # bias
        self.weights[0] += self.learning_rate * error
        # weights
        self.weights[1] += self.learning_rate * error * x[0]
        return error
    
    def train(self, X, Y):
        for epoch in range(self.n_epoch):
            error_epoch = 0
            print("\nepoch:", epoch)
            for i, x in enumerate(X):
                y_expected = Y[i]
                error = self.learn(x, y_expected)
                error_epoch += abs(error)
            if (error_epoch / 2) < self.error_min:
                break
                
    def output(self, inputs):
        weighted_sum = self.calculate_weighted_sum(inputs)
        return self.activation_function(weighted_sum)

    





