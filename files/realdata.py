import math

mD = 9.869233e-13/1000
k = 1*mD
rho = 1080
g = 9.81
mhu = 0.46*1e-3

K = (k*rho*g)/mhu
print("hydraulic conductivity is", K)