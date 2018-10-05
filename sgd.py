import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)

data = load_boston()
sclr = StandardScaler()
df = pd.DataFrame(sclr.fit_transform(data.data), columns=data.feature_names)

# df = sclr.fit_transform(df)
# print(df)
df['price'] = data.target

data = df.values


# data.shape

def MSE(y: np.array, pred: np.array) -> float:
    """
    It finds the mean squared error for
    the converged intercept and weight
    vector using the dataset. Mostly the
    given dataset should be cross-validation
    dataset to reduce the overfitting. The
    following implementation was kept vectorized
    so that we can use amazing speed of numpy.
    """

    assert (y.shape[0] == pred.shape[0])
    N = pred.shape[0]
    y = y.reshape(N, 1)
    pred = pred.reshape(N, 1)
    residuals = y - pred
    squared_errors = np.power(residuals, 2)
    total_squared_errors = np.sum(squared_errors, axis=0)
    mse = (1.0 / N) * total_squared_errors
    return round(mse[0], 5)


def predict(X, W, b):
    N, d = X.shape
    W = W.reshape(d, 1)
    return (X @ W) + b


def SGD(training_frame,
        validation_frame=None,
        use_validation_error=False,
        random_state=42,
        loss='squared_loss',
        tol=1e-3,
        learning_rate='adaptive',
        eta0=0.01,
        early_stopping=True,
        no_iter_change=5,
        max_iters=1000,
        batch_size=100,
        verbose=False,
        **kwargs):
    """
    Non-regularized SGD for Linear Regression.
    SGD stands for Stochastic Gradient Descent: the gradient
    of the loss is estimated each sample at a time
    and the model is updated along the way with a
    decreasing strength schedule (aka learning rate).

    Parameters
    ----------

    X: np.ndarray:
        The feature set of the datapoints

    y: np.array:
        The Response variable

    loss: str, default: 'squared_loss'
        The loss function to be used

    tol: float or None, default: 1e-3
        Tolerance level of the stopping criteria

    random_state: int
        The random_state to be used for random seed

    learning_rate: str, default: 'adaptive'
        The type of learning rate.
        If 'adaptive': make eta0 = eta0 / 5 if new
                       error > old_error - tol
        If 'constant': Don't change eta0

    eta0: float, default: 0.01
    """

    # Define the initial W
    np.random.seed(random_state)
    multiplier = np.random.randint(low=5)
    np.random.seed(random_state)
    W = np.random.random(size=data.shape[1] - 1) * multiplier
    W = W.reshape(data.shape[1] - 1, 1)

    # Define the intial intercept
    b = 0

    # pW = None
    # pB = None
    pE = np.Inf  # Set the initial error as infinite

    if validation_frame is not None:
        cv_X = validation_frame[:, 0:-1]
        cv_y = validation_frame[:, -1]

    for each_iter in range(1, max_iters):

        if verbose:
            print(f'======= Converging for iter {each_iter} =======')

        np.random.seed(random_state)
        sample = training_frame[np.random.choice(training_frame.shape[0], batch_size, replace=False), :]

        X = sample[:, 0:-1]
        y = sample[:, -1]
        N, d = X.shape
        y = y.reshape(N, 1)

        # Store the previous weights and intercept
        pW = W.reshape(d, 1)
        pB = b

        # Find the derivatives of the weight and intercept

        dowW = y - predict(X, pW, pB)
        dW = X.T @ dowW
        dW = (-2.0 / N) * dW
        dB = (-2.0 / N) * np.sum(dowW)

        # Update the Weights and intercept

        W = pW - (eta0 * dW)
        b = pB - (eta0 * dB)

        #########################################
        pred = predict(X, W, b)
        if validation_frame is not None:
            cv_pred = predict(cv_X, W, b)

        error = pE
        if loss == 'squared_loss':
            training_error = MSE(y, pred)
            if validation_frame is not  None:
                cv_error = MSE(cv_y, cv_pred)

            if use_validation_error and validation_frame is not None:
                error = cv_error
            else:
                error = training_error
        # Add some other losses

        if early_stopping:
            if error >= pE - tol and no_iter_change==0:
                if verbose:
                    print('Can not converge futher')
                    print(f'Trining error {pE}')
                    if use_validation_error and cv_error:
                        print(f'Trining error {cv_error}')
                return {
                    'eta': eta0,
                    'W': W,
                    'b': b,
                    'iter': each_iter,
                    'train_MSE': pE,
                    'CV_MSE': None if not cv_error else cv_error
                    }
            elif error >= pE - tol and no_iter_change != 0:
                no_iter_change -= 1
                if learning_rate == 'adaptive':
                    eta0 = eta0 / 5
                    if verbose:
                        print(f'Adaptive learning rate..New eta0 = {eta0}')
        elif learning_rate == 'adaptive' and not early_stopping and error >= pE - tol:
            if learning_rate == 'adaptive':
                eta0 = eta0 / 5
                if verbose:
                    print(f'Adaptive learning rate..New eta0 = {eta0}')

        pE = error
        if verbose:
            print(f'\n\tTraining error: {error} \n')


training_frame = data[0:350]
validation_frame = data[351:]
print(SGD(training_frame=training_frame, validation_frame=validation_frame,
          random_state=42, verbose=True, max_iters=10000, learning_rate='adaptive', use_validation_error=False))



from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(training_frame[:, 0:-1], training_frame[:, -1])
pred = clf.predict(validation_frame[:, 0:-1])
print(MSE(validation_frame[:, -1], pred))
