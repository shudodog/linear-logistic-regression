# linear-logistic-regression without scikit-learn
**2021-October**

**Python. (I didn't use built-in regression functions)**

##All datas in txt File

### Linear Regression with Multiple Features
* the selling price of a house and multiple factors(11)
* I used linear regression the gradient descent and the normal equation
* I tried different learning rate and visualized in gradient descent.
* I got the result of house price using normal equation(45.89) and gradient descent(52.11).

## the gradient descent function and visualized image
```
def normalz(x):
    mu = np.mean(x, axis = 0)
    sig = np.std(x, axis= 0, ddof = 1)
    norm = (x - mu)/sig
    return norm, mu, sig

def cost_1(X, y, theta):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    return 1/(2 * m) * errors.T.dot(errors)

def gradient(X, y, theta, learning_rate, iterations):
    ccost = np.zeros(iterations)
    for i in range(iterations):
        prd = X.dot(theta)
        error = np.subtract(prd, y)
        delta = (learning_rate / m) * X.transpose().dot(error);
        theta = theta - delta;
        ccost[i] = cost_1(X, y, theta)
    return theta, ccost
```

![1](https://user-images.githubusercontent.com/76150392/139625907-c08f5069-fd6a-4ea9-96c1-4170b70f894f.GIF)

![2](https://user-images.githubusercontent.com/76150392/139625922-c0fd4ffc-c316-47cb-900b-0bd0fa49f923.GIF)

![3](https://user-images.githubusercontent.com/76150392/139625955-18a64025-d3f6-45b8-b0c8-92f31b9df29f.GIF)


### Logistic Regression
* national test scores for students who applied for a certain university and las column indicated admitted result.
* My logistic regression model's performance is 88.89%.
* I tried visualize the decision boundary.
* I got the result of each students of predict admitted result(b,d student are admiteed and a, c student are not).

## Logistic Regression function and visualized image
```
from scipy.optimize import fmin_tnc as fm
class LogisticRegression:
    def cost(self, theta, x, y):
        m = x.shape[0]
        return  -(1 / m) * np.sum(np.log(sig(np.dot(x, theta)))*y +  np.log(1 - sig(np.dot(x, theta)))*(1 - y) )

    def gradient(self, theta, x, y):
        return (1 / x.shape[0]) * np.dot(x.T, sig(np.dot(x, theta)) - y)

    def fit(self, x, y, theta):
        w = fm(func=self.cost, x0=theta, fprime=self.gradient, args=(x, y.flatten()))
        self.w = w[0]
        return self

    def predict(self, x):
        theta = self.w[:, np.newaxis]
        return sig(np.dot(x, theta))

    def accuracy(self, x, test):
        theta = self.w[:, np.newaxis]
        prdc = (sig(np.dot(x, theta)) >= 0.5).astype(int)
        prdc = prdc.flatten()
        return 100*np.mean(prdc == test)
```

![4](https://user-images.githubusercontent.com/76150392/139626300-4c7b2593-0cae-4531-9e64-f359e804c621.GIF)

## What I've learned
* Basic theory of Logistic Regression and Linear Regression
* different learning rates effect on the gradient descent
* Coding without built in functions



