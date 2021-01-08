import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# height (cm)
X = np.array([[
    98.508882848198,71.2530701092942,53.0377224247045,44.181444068749,38.0525951808809,33.0605505096331,29.0172362570938,
    97.86214794291,69.5844810284592,56.5685424949238,46.5295604965274,39.6232255123179,32.5576411921994,30.2654919008431,
    100.498756211209,67.1863081289633,52.2398315464359,43.2897216438267,37.0540146272978,32.1403173599764,29.0688837074973,
    87.692645073575,66.4830805543786,52.3450093132096,42.190046219458,36.3455636907725,31.1448230047949,27.0739727413618,
    97.1648084442099,67.3646791723972,56.1426753904728,45.0444225182208,15.2970585407784,31.0644491340181,28.0713376952364,
    92.1954445729289,64.007812023221,52.0864665724217,45,36,31,27.0185121722126,
    87.0229854693575,62.072538211354,49.0917508345343,42.1070065428546,35.0570962859162,31.0644491340181,28.0178514522438
]]).T
# weight (kg)
X = 1/X
y = np.array([[70, 100, 130,  160, 190, 220, 250, 70, 100, 130,  160, 190, 220, 250, 70, 100, 130,  160, 190, 220, 250, 70, 100, 130,
               160, 190, 220, 250, 70, 100, 130,  160, 190, 220, 250, 70, 100, 130,  160, 190, 220, 250, 70, 100, 130,  160, 190, 220, 250]]).T
print(X.shape)
print(y.shape)
# Visualize data
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# w_0 = w[0][0]
# w_1 = w[1][0]
# x0 = np.linspace(145, 185, 2)
# y0 = w_0 + w_1*x0



# fit_intercept = False for calculating the bias
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)
print('Solution found by scikit-learn  : ', regr.coef_)
# print('Solution found by (5): ', w.T)

# plt.plot(X.T, y.T, 'ro')     # data
# plt.plot(x0, y0)               # the fitting line
# plt.axis([0, 0.05, 60, 260])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()
# print(Xbar)
plt.plot(X, y, 'ro')
plt.axis([0, 0.05, 60, 260])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
