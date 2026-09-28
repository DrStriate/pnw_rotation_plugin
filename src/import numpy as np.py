import numpy as np
# from scipy import linalg


def weighted_inversion(A, b, W):
  """
  Solves Ac = b using Weighted Least Squares.
  Returns optimal parameters c.
  """
  # 1. Compute the components for the weighted normal equation
  AT = A.T
  ATA = AT.dot(W).dot(A)
  ATb = AT.dot(W).dot(b)

  # 2. Solve the linear system directly instead of inverting,
  # which is computationally more stable.
  # Solves (AT * W * A) * c = (AT * W * b)
  # c = linalg.solve(ATA, ATb)

  c, residuals, rank, s = np.linalg.lstsq(ATA, ATb, rcond=None)
  return c


# --- Example Usage ---
# Define the system: 3 observations, 2 parameters
A = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

b = np.array([3.0, 7.0, 11.0])

# Define weights (e.g., 3 observations with different confidence)
weights = np.array([1.0, 0.5, 0.1])
W = np.diag(weights)

# Run inversion
c_estimated = weighted_inversion(A, b, W)
print("Estimated parameters c:", c_estimated)
