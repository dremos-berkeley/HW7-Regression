"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
	model = LogisticRegressor(num_feats=2, learning_rate=0.01)

	# set weights to zeros so sigmoid should output 0.5
	model.W = np.array([0.0, 0.0, 0.0])

	X = np.array([[1, 2, 1], [3, 4, 1]])
	pred = model.make_prediction(X)

	# should be 0.5 for all when weights are zero
	assert np.allclose(pred, [0.5, 0.5])

	# also check output is between 0 and 1
	model.W = np.array([1.0, -1.0, 0.5])
	pred = model.make_prediction(X)
	assert all(p >= 0 and p <= 1 for p in pred)

def test_loss_function():
	model = LogisticRegressor(num_feats=2)

	y_true = np.array([1, 0, 1, 0])

	# good predictions
	y_pred_good = np.array([0.99, 0.01, 0.99, 0.01])
	loss1 = model.loss_function(y_true, y_pred_good)

	# bad predictions (flipped)
	y_pred_bad = np.array([0.01, 0.99, 0.01, 0.99])
	loss2 = model.loss_function(y_true, y_pred_bad)

	# good preds should have lower loss than bad preds
	assert loss1 < loss2

	# loss should be positive
	assert loss1 >= 0

def test_gradient():
	model = LogisticRegressor(num_feats=2)
	model.W = np.zeros(3)

	X = np.array([[1, 0, 1], [0, 1, 1]])
	y_true = np.array([1, 0])

	grad = model.calculate_gradient(y_true, X)

	# check shape matches weights
	assert grad.shape == model.W.shape

	# manually compute expected gradient
	# when W=0, sigmoid gives 0.5, so errors are [-0.5, 0.5]
	errors = np.array([0.5 - 1, 0.5 - 0])
	expected = (1/2) * X.T @ errors
	assert np.allclose(grad, expected)

def test_training():
	np.random.seed(42)  # for reproducibility
	model = LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=10, batch_size=2)

	W_before = model.W.copy()

	# dummy data
	X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
	y_train = np.array([0, 0, 1, 1])
	X_val = np.array([[2, 3], [6, 7]])
	y_val = np.array([0, 1])

	model.train_model(X_train, y_train, X_val, y_val)

	# make sure weights actually changed
	assert not np.allclose(model.W, W_before)

	# check loss history got filled
	assert len(model.loss_hist_train) > 0
	assert len(model.loss_hist_val) > 0
