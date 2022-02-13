import numpy as np


def dx1x1(a):
    return -2*a**2


def dx2x2(a):
    return -2*a**2+a**3


def dx1x2(a):
    return -2*a**2+16+a**3


def Hes(a):
    return dx1x1(a)*dx2x2(a)-(dx1x2(a))**2


a_list = [-2]
for a in a_list:
    print("-"*20)
    print("dx1x1:{: >35}".format(dx1x1(a)))
    print("dx2x2:{: >35}".format(dx2x2(a)))
    print("dx1x2:{: >35}".format(dx1x2(a)))
    print("Hess:{: >35}".format(Hes(a)))
    print("-"*20)
    print("\n")
