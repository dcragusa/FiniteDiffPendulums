# Computational Physics, Project A
# David Christopher Ragusa

import numpy as np
import matplotlib.pyplot as plt

#
# Single Pendulum System
#

D = 0.0 # change for damping
h = 0.1

L = np.matrix([[0.0, 1.0],
               [-1.0,-D ]])

y0s = np.matrix([[0.1],
                 [0.0]])


def euler(steps, y0=y0s, L=L, h=h):
  'Implementation of the Euler method.'
  results = [y0]
  T = np.identity(2) + h*L
  y = y0
  for i in xrange(steps): # xrange faster for iteration
    results.append(T*y)
    y = T*y
  return results


def leapfrog(steps, y0=y0s, L=L, h=h):
  'Implementation of the Leapfrog method.'
  results = [y0]
  ystored = impliciteuler(1)[-1]
  yminus1 = y0
  y = ystored
  results.append(y)
  for i in xrange(steps-1):
    yplus1 = yminus1 + 2*h*L*y
    results.append(yplus1)
    y = yplus1
    yminus1 = ystored
    ystored = y
  return results


def rk4(steps, y0=y0s, L=L, h=h):
  'Implementation of the RK4 method.'
  results = [y0]
  y = y0
  for i in xrange(steps):
    k1 = h*L*y
    k2 = h*L*(y+0.5*k1)
    k3 = h*L*(y+0.5*k2)
    k4 = h*L*(y+k3)
    y = y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    results.append(y)
  return results


def impliciteuler(steps, y0=y0s, L=L, h=h):
  'Implementation of the Implicit Euler method.'
  results = [y0]
  temp = np.identity(2) - h*L
  T = temp.I # inverse
  y = y0
  for i in xrange(steps):
    y = T*y
    results.append(y)
  return results


def energysing(y):
  'Returns the energy of the single pendulum system.'
  theta, dottheta = np.ravel(y) # ravel turns matrix into a list - easier assignment
  return 0.5*dottheta**2 + 1 - np.cos(theta)


def stability():
  '''
  Searches for the critical step size for the single pendulum system.
  Assumes the critical step size is below h = 3, and counts down 0.1
  at a time. When the boundary is found the function iterates around
  that boundary until h is accurate to 10 d.p.
  '''
  h = 3.0
  diff = 0.1
  direction = 's'
  while True:
    if diff < 1e-9 and h > 3:
      return 'unconditionally stable'
    if h <= 1e-3: # too small to be practical
      return 'unconditionally unstable'
    if diff < 1e-9:
      return h  # output
    steps = (0 if h <= 0 else int(100.0/h)) # constant 100 time units
    methodresults = rk4(steps, h=h) # modify this line for different functions
    if np.isnan(energysing(methodresults[-1])): # blew up
      unstable = True
    elif h == 0:
      unstable = False
    else:
      maxenergy = np.amax([energysing(i) for i in methodresults])
      unstable = maxenergy > 2*energysing(methodresults[0])
    if unstable:
      if direction == 'l':
        direction == 's' # always home downwards
      h -= diff
    else:
      h += diff
      if direction == 's':
        diff /= 10
        h -= diff


def rk4graph():
  'Plots Fig. 1 in the report.'
  bigh = 2
  bigsteps = 10
  smallh = 0.1
  smallsteps = 200
  xplot1 = [i*bigh for i in xrange(bigsteps+1)]  # I love list comprehensions
  xplot2 = [i*smallh for i in xrange(smallsteps+1)]
  yplot1 = [i.item(0) for i in rk4(bigsteps, h=bigh)]
  yplot2 = [i.item(0) for i in rk4(smallsteps, h=smallh)]
  fig, ax = plt.subplots()
  ax.plot(xplot1, yplot1, 'r', label='RK4, h=2')
  ax.plot(xplot2, yplot2, 'g', label='RK4, h=0.1')
  ax.set_xlabel('Dimensionless Time')
  ax.set_ylabel('Theta')
  ax.set_ylim(-0.08, 0.11)
  ax.legend(loc='upper right', shadow=True)
  plt.show()

#
# Double Pendulum System
#

R = 1.0 #
G = 0.0 # change for the graphs

Ld = np.matrix([[  0.0,      0.0,        1.0,        0.0 ],
                [  0.0,      0.0,        0.0,        1.0 ],
                [-(R+1.0),    R,         -G,         0.0 ],
                [(R+1.0), -(R+1.0), G*(1.0-1.0/R), -(G/R)]])

y0d = np.matrix([[0.1],
                 [0.0],
                 [0.0],
                 [0.0]])


def energydoub(y, R=R):
  'Returns the energy of the double pendulum system.'
  theta, phi, dottheta, dotphi = np.ravel(y)
  return 0.5*(dottheta**2+R*(dottheta**2+dotphi**2+dottheta*dotphi))+1-np.cos(theta)+R*(2-np.cos(theta)-np.cos(phi))


def graphdoubenergy():
  'Plots graph of energy against time for the double pendulum system.'
  h = 0.1
  steps = int(100.0/h)
  methodresults = rk4(steps, y0=y0d, L=Ld, h=h)
  energydoub(methodresults[0])
  xplot = [i*h for i in xrange(steps+1)]
  yplot = [energydoub(i) for i in methodresults]
  fig, ax = plt.subplots()
  ax.plot(xplot, yplot, 'r')
  ax.set_xlabel('Dimensionless Time')
  ax.set_ylabel('Dimensionless Energy')
  plt.show()


def graphdoubangles():
  'Plots graph of angular motion against time for the double pendulum system.'
  h = 0.1
  steps = int(50.0/h)
  methodresults = rk4(steps, y0=y0d, L=Ld, h=h)
  xplot = [i*h for i in xrange(steps+1)]
  yplot = [i.item(0) for i in methodresults]
  yplot2 = [i.item(1) for i in methodresults]
  fig, ax = plt.subplots()
  ax.plot(xplot, yplot, 'r', label="Theta")
  ax.plot(xplot, yplot2, 'g', label="Phi")
  ax.set_xlabel('Dimensionless Time')
  ax.set_ylabel('Angles')
  ax.legend(loc='upper right', shadow=True)
  plt.show()
