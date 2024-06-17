#!/usr/bin/env python3

# Compute coefficients for the constrained optimization quadratic problem.
#     @param D Gravity in our sensor frame
#     @param d Rotation from the arbitrary inertial reference frame to this gravity vector
#     @param gravity_mag Scalar size of gravity (normally is 9.81)
#     @return Coefficents from highest to the constant

# matlab version: https://gist.github.com/goldbattle/3791cbb11bbf4f5feb3f049dad72bfdd
'''
D = sym('D',[3,3])
d = sym('d',[3,1])
lambda = sym('lambda');
g = sym('g');

assume(D,'real')
assume(d,'real')
assume(lambda,'real')
assume(g,'real')


expression = det((D - lambda*eye(3,3))^2 - (1/g^2)*(d*d'));
collected = collect(expression, lambda)
'''

import sympy as sp

def SymbolMatrix(symbol, shape, **kwargs):
  rows = list()
  for i in range(shape[0]):
    row = list()
    if shape[1] == 1:
      # ele_symbol = symbol + '_' + str(i) + '_' + str(j)
      ele_symbol = symbol + str(i+1)
      row.append(sp.Symbol(ele_symbol, **kwargs))
    else:
      for j in range(shape[1]):
        # ele_symbol = symbol + '_' + str(i) + '_' + str(j)
        ele_symbol = symbol + str(i+1) + '_' + str(j+1)
        row.append(sp.Symbol(ele_symbol, **kwargs))
    rows.append(row)
  return sp.Matrix(rows)

if __name__ == '__main__':

  D = SymbolMatrix('D', (3,3), real=True)
  d = SymbolMatrix('d', (3,1), real=True)
  lambda_ = sp.Symbol('lambda', real=True)

  # # ↓ Why it yeilds zero coeffs if we use this?
  # g = sp.Symbol('g', real=True, positive=True, nonzero=True)
  # # expression = sp.det((D - lambda_ * sp.eye(3,3))**2 - ((1/g)**2)*(d*d.T))
  # g_inv = (1/g)
  # expression = sp.det((D - lambda_ * sp.eye(3,3))**2 - (g_inv**2)*(d*d.T))

  g_inv = sp.Symbol('g_inv', real=True)
  expression = sp.det((D - lambda_ * sp.eye(3,3))**2 - (g_inv**2)*(d*d.T))

  # print(expression)
  collected = sp.collect(expression, lambda_)
  # print(collected)
  # print(type(collected))

  coeff0 = collected.coeff(lambda_, 0)
  coeff1 = collected.coeff(lambda_, 1)
  coeff2 = collected.coeff(lambda_, 2)
  coeff3 = collected.coeff(lambda_, 3)
  coeff4 = collected.coeff(lambda_, 4)
  coeff5 = collected.coeff(lambda_, 5)
  coeff6 = collected.coeff(lambda_, 6)

  print("coeff0: {}\n".format(coeff0))
  print("coeff1: {}\n".format(coeff1))
  print("coeff2: {}\n".format(coeff2))
  print("coeff3: {}\n".format(coeff3))
  print("coeff4: {}\n".format(coeff4))
  print("coeff5: {}\n".format(coeff5))
  print("coeff6: {}\n".format(coeff6))

