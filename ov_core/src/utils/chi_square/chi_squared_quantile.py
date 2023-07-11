#!/usr/bin/env python3

import os
from scipy.stats import chi2


x = 0.95
N = 2000
table_name = "chi_squared_quantile_table_0_95"

#print(chi2.ppf(x, 0))
print(chi2.ppf(x, 1))
print(chi2.ppf(x, 2))
print(chi2.ppf(x, 3))


s = "const double {}[] = ".format(table_name) + "{\n";
s += "\t-1.0,\t\t// dof=0\n"
for i in range(N):
    if i < N - 1:
        s += "\t{},\t\t// dof={}\n".format(chi2.ppf(x, i+1), i+1)
    else:
        s += "\t{}\t\t// dof={}\n".format(chi2.ppf(x, i+1), i+1)
        s += "};\n"
print(s)
with open("{}.cpp".format(table_name), "w") as stream:
  stream.write(s)
with open("{}.h".format(table_name), "w") as stream:
  stream.write("extern const double {}[];\n".format(table_name))



