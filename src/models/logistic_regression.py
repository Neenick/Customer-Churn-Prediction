from sklearn.linear_model import LogisticRegression



class BaseModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, max_iter=10000)

    def train(self, X, y):
        self.model.fit(X, y)
        

    def evaluate(self, X, y):
        print(f"Test Accuracy {self.model.score(X, y)}")