from PerceptronNot import PerceptronNot

class Perceptron:
    def __init__(self, learning_rate, n_epoch, error_min):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.error_min = error_min
        self.weights = [0, 0, 0] # bias, weight1, weight2
    
    def activation_function(self, weighted_sum):
        # step function
        if weighted_sum > 0:
            ret = 1.0
        else:
            ret = 0 
        return ret
    
    def calculate_weighted_sum(self, inputs):
        weighted_sum = self.weights[0]
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i + 1]
        
        return weighted_sum

    def learn(self, x, y_expected):
        y_calculated = self.output(x)
        error = y_expected - y_calculated
        print(f"Input: {x}, y_expected: {y_expected}, error: {error}, weights: {self.weights}")
        # bias
        self.weights[0] += self.learning_rate * error
        # weights
        for i in range(len(self.weights) - 1):
            self.weights[i + 1] += self.learning_rate * error * x[i]
        return error
        
    def train(self, X, Y):
        for epoch in range(self.n_epoch):
            error_epoch = 0
            print("\nepoch:", epoch)
            for i, x in enumerate(X):
                y_expected = Y[i]
                error = self.learn(x, y_expected)
                error_epoch += abs(error)
            if (error_epoch / 4) < self.error_min:
                break
                
    def output(self, inputs):
        weighted_sum = self.calculate_weighted_sum(inputs)
        return self.activation_function(weighted_sum)


# AND Table
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_and = [0, 0, 0, 1]

# OR Table
X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_or = [0, 1, 1, 1]

# NOT Table
X_not = [[0], [1]]
Y_not = [1, 0]

# not perceptron 
print("---- NOT TRAINING ----")
perceptron_not = PerceptronNot(learning_rate=0.01, n_epoch=100, error_min=0.1)
perceptron_not.train(X_not, Y_not)

# and perceptron
print("\n---- AND TRAINING ----")
perceptron_and = Perceptron(learning_rate=0.01, n_epoch=100, error_min=0.1)
perceptron_and.train(X_and, Y_and)

# and perceptron
print("\n---- OR TRAINING ----")
perceptron_or = Perceptron(learning_rate=0.01, n_epoch=100, error_min=0.1)
perceptron_or.train(X_or, Y_or)

print(f"\nAND Perceptron: Bias: {perceptron_and.weights[0]}, weights: {perceptron_and.weights[1:3]}")    
print(f"OR Perceptron: Bias: {perceptron_or.weights[0]}, weights: {perceptron_or.weights[1:3]}")    
print(f"NOT Perceptron: Bias: {perceptron_not.weights[0]}, weights: {perceptron_not.weights[1]}")   

# testing outputs
inputs = [0, 0]
for i in range(4):
    inputs = [i // 2, i % 2]

    or_output = perceptron_or.output(inputs)
    and_output = perceptron_and.output(inputs)
    not_output = perceptron_not.output([and_output])
    
    print(f"\n{inputs}") 
    ex_or_output = perceptron_and.output([or_output, not_output])   
    print(f"OUTPUT: {ex_or_output}")   




    





