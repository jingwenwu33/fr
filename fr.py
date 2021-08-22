import streamlit as st
import pandas as pd
import math
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

st.title('单管冻结温度场冻结半径估测')
r0=st.number_input('冻结管半径值：（单位m）')
tct=st.number_input('冻结管温度值：（单位℃）')
r1=st.number_input('测温点半径值：（单位m）')
tr1=st.number_input('测温点温度值：（单位℃）')

def radius(x0,x1,t0,t1):
  c1 = (t1 - t0) / (math.log(x1) - math.log(x0))
  c2 = t1 - (t1 - t0) * math.log(x1) / (math.log(x1) - math.log(x0))
  result= (math.e) ** (-c2 / c1)
  return result


if st.button('计算冻结半径'):
  st.write('根据**_特鲁巴克单管温度场分布公式 _**可得，\n冻结半径为',radius(r0,r1,tct,tr1),'m')

if st.button('获取数据总表'):
  st.write(pd.DataFrame({'半径值/m':[r0,r1,radius(r0,r1,tct,tr1)],'温度值/℃':[tct,tr1,0]}))

if st.button('冻土温度云图'):
  def f(x, y, data):
    r1 = data['r1']
    r0 = data['r0']
    tct = data['tct']
    tr1 = data['tr1']
    ybsn = data['ybsn']

    r = (x ** 2 + y ** 2) ** 0.5
    fenzi = np.log((1 / r) * float(ybsn))
    fenmu = float(np.log(float(ybsn) / r0))
    return tct * (fenzi / fenmu)


  x, y, t, b, a = symbols('x y t b a')
  fx = t * ln(a / x) / ln(a / b) - y
  list_f1 = solve(fx, a)
  a1 = ''
  for i in list_f1:
    a1 = a1 + str(i)
  a2 = sympify(a1)
  a3 = a2.evalf(subs={x: r1, y: tr1, t: tct, b: r0})
  a3 = ('%.6f' % a3)
  data = {'r1': float(r1), 'r0': float(r0), 'tct': float(tct), 'tr1': float(tr1), 'ybsn': float(a3)}

  x = np.linspace(-5, 5, 180)
  y = np.linspace(-5, 5, 180)
  X, Y = np.meshgrid(x, y)

  cset = plt.contourf(X, Y, f(X, Y, data), 15, cmap=plt.cm.jet)  # YlGnBu
  contour = plt.contour(X, Y, f(X, Y, data), 15, colors='k')
  plt.clabel(contour, fontsize=15, colors='k')

  cbar = plt.colorbar(cset)

  r = float(data['ybsn'])
  a, b = (0., 0.)
  theta = np.arange(0, 2 * np.pi, 0.01)
  x_c = a + r * np.cos(theta)
  y_c = b + r * np.sin(theta)
  plt.plot(x_c, y_c, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=5)

  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
