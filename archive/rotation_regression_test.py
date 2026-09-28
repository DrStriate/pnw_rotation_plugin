import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
  labels = []
  R = []
  T = []

  source = input(
      "Enter source: 1 - Wells Simpson, 2 - large set, 3 - quadratic test")
  if source == "1":
    print("parameters from csv from Wells Simpson paper (edited)")
    file_path = "data/WellsSimpson2001.csv"
    title = file_path
    print(f"Selected file: {file_path}")
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for row in reader:
        if row["R"] != "" and row["T"] != "" and row["Use"] == "Y":
          R.append(float(row["R"]))
          T.append(float(row["T"]))
          labels.append(row["Tag"])

  if source == "2":
    print("test parameters from csv using published paleo")
    file_path = "data/Large paleo rot set.csv"
    title = file_path
    print(f"Selected file: {file_path}")
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for row in reader:
        if row["Rotation_deg"] != "" and row["Age_Ma"] != "" and int(row["Quality"]) > 0:
          R.append(float(row["Rotation_deg"]))
          T.append(float(row["Age_Ma"]))

  if source == "3":
    title = "quadratic fit test"
    print("test parameters from quadratic fit to graph:")
    a = -0.0088  # from quadratic fit
    b = 0.4766  # from quadratic fit
    w_0 = 1.19  # from Wells - Simpson Fig3B
    print("a = ", a)
    print("b = ", b)
    print("w_o = ", w_0)
    RT = createTestDataSet(a, b, w_0, 50, 10)

  # print ("RT: ", RT)
  c = calculateRateCoeff(R, T, normalize=False)
  print("coeffeciants c: ", c)
  Erms = calculateRegressionRmsError(c, R, T)
  print("Calibration residual rms: ", Erms)

  # Create the plots
  # Paleomag data plot
  fig, ax1 = plt.subplots()
  ax1.scatter(T, R, label='Paleomag Data', color='blue')
  ax1.set_ylim(0)
  for i, txt in enumerate(labels):
    pad = 1
    ax1.annotate(txt, (T[i] + pad, R[i] + pad))
  ax1.set_ylabel('Total Rotation (Deg)', color='tab:blue')
  ax1.set_xlabel('T (Ma)')
  ax1.tick_params(axis='y', labelcolor='tab:blue')

  # quadratic coefficients speed plot
  ax2 = ax1.twinx()
  T = np.linspace(0, 50, 11)
  w = calculateRotationRateFromCoeff(c, T)
  ax2.plot(T, w, color='orange', linestyle='--')
  ax2.set_ylim(0)
  ax2.set_xlabel('T (Ma)')
  ax2.set_ylabel('Quadratic Rot Rate (Deg / Ma)', color='tab:orange')

  # quadratic recalculation of total rotation
  Rc = calculateTotalRotationFromCoeff(c, T)
  ax1.plot(T, Rc, color='blue', linestyle='-')

  # calculate average rate over 50 Ma
  meanRate = sum(w) / len(w)
  print("Mean rate = ", meanRate)

  # 3. Add details
  plt.title("Rotation Rate Curve From " + title)
  plt.grid(True)

  # 4. Display or save
  plt.show()  # Opens a window to show the plot
  # plt.savefig('my_plot.png') # Saves to file

  return


def col(matrix, colNo):
  return [row[colNo] for row in matrix]


def calculateRateCoeff(Ri, Ti, normalize=False):
  Xi = []
  for i in range(len(Ti)):
    Xi.append([Ti[i] * Ti[i] * Ti[i] / 3.0, Ti[i] * Ti[i] / 2.0, Ti[i]])

  X = np.array(Xi)
  Y = np.array(Ri)

  A = X.T @ X
  b = X.T @ Y

  x = np.linalg.solve(A, b)

  if (normalize):
    x = x / x[2]

  return x


def calculateRotationRateFromCoeff(c, Tary):
  w = []
  for T in Tary:
    w.append(pow(T, 2) * c[0] + T * c[1] + c[2])
  return w


def calculateTotalRotationFromCoeff(c, Tary):
  R = []
  for T in Tary:
    R.append((float)(pow(T, 3) * c[0] / 3.0 +
             pow(T, 2) * c[1] / 2.0 + T * c[2]))
  return R


def calculateRegressionRmsError(c, R, T):
  Rc = calculateTotalRotationFromCoeff(c, T)
  N = len(T)
  sum = 0
  for i in range(N):
    e = R[i] - Rc[i]
    sum = e * e
  rms = np.sqrt(sum) / N
  return rms


def createTestDataSet(a, b, w_0, tMax, N):
  RT = []
  deltaT = tMax / N
  for i in range(N):
    T = i * deltaT
    R = (1.0/3.0 * a * pow(T, 3) + 0.5 * b * pow(T, 2) + T) * w_0
    RT.append([R, T])
  return RT


if __name__ == "__main__":
  # The return value of main() is passed to sys.exit()
  # to provide a proper exit code to the operating system.
  sys.exit(main())
