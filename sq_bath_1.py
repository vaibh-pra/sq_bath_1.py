#matplotlib inline
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
#import math
from IPython.display import Image
from qutip import *
import sympy as sy
from sympy import I
from math import *
from sq_bath_func import *


#System Parameters (in units of wm)
#-----------------------------------
Nc = 2                     # Number of cavity states
Nm = 3                     # Number of mech states
kappa = 0.05               # Cavity damping rate
E = 0.005                  # Driving Amplitude         
g0 = 0.8                   # Coupling strength
Qm = 1e4                   # Mech quality factor
gamma = kappa/3            # Mech damping rate



tlist = np.linspace(0, 200, 1000)
taulist = tlist * kappa
delta = 0.0


# Initialize1
#-------------------
print('\n 0-->optical\n 1-->mechanical')
c1 = int(input())
if c1==0:
    print(u'Input intial state: (|n1> + |n2>) \u2297 |m>')
    n1 = int(input())
    n2 = int(input())
    m = int(input())
    print('|n1>=|%i>,'%n1, '|n2>=|%i>,'%n2, '|m>=|%i>'%m)
    rho0 = initst(Nc, Nm, n1, n2, m, c1)
if c1==1:
    print(u'Input intial state: |n> \u2297 (|m1> + |m2>)')
    m1 = int(input())
    m2 = int(input())
    n = int(input())
    print('|n>=|%i>,'%n, '|m1>=|%i>,'%m1, '|m2>=|%i>'%m2)
    rho0 = initst(Nc, Nm, m1, m2, n, c1)

d = tensor(destroy(Nc), qeye(Nm))
c = tensor(qeye(Nc), destroy(Nm))
num_d = d.dag()*d
num_c = c.dag()*c


H_nonlin = 10**3 * (num_d) + num_c - (g0 * num_d)**2

alpha = sol(gamma, kappa, delta, g0, E)[0]
alpha_s = sol(gamma, kappa, delta, g0, E)[1]
b = sol(gamma, kappa, delta, g0, E)[2]
deltaP = delta - g0 * b


H_nonlinear = -deltaP*num_d+num_c+g0*(complex(alpha_s)*d+complex(alpha)*d.dag())*(c+c.dag())+g0*num_d*(c+c.dag())


t, t1, th = [], [], []
thetalist = np.linspace(0, 1, 100)

print('Input the value of thermal occupation number n_th')
n_th = float(input())

print('Input r value between 0 and 1')
rs = float(input())

print('\n 1 --> SME\n 11 --> squeezed SME\n 2 --> thermal dephased-SME\n 21 --> squeezed dephased-SME\n 3 --> thermal DSME\n 31 --> squeezed DSME')
x = int(input())
y = int(input())
strx = legstr(x)
stry = legstr(y)

print(u'Input q1, q2 for matrix element norm |<q1| \u03f1 (t) |q2>|')
q1 = int(input())
q2 = int(input())

#Calculate1
#---------------
for i in range(len(thetalist)):
    theta = i * 0.01 * 2*pi  
    
    result0 = mesolve(H_nonlin, rho0, taulist, collapse(n_th, rs, theta, x,g0), [])
    for k in range(len(result0.times)):
        mat_ele0 = np.array(ptrace(result0.states[k],c1))[q1][q2]
        rho_evol0 = LA.norm(mat_ele0)
        if rho_evol0 < 0.1:
            t.append(result0.times[k])
            break
    
    
    result1 = mesolve(H_nonlin, rho0, taulist, collapse(n_th, rs, theta, y,g0), [])
    for k in range(len(result1.times)):
        mat_ele1 = np.array(ptrace(result1.states[k],c1))[q1][q2]
        rho_evol1 = LA.norm(mat_ele1)
        if rho_evol1 < 0.1:
            t1.append(result1.times[k])
            break
    th.append(theta/(2*pi))
    


#Initialize2
#-----------------
print(u'Input \u0398/2\u03c0 value between 0 and 1')
thet = float(input())
theta = thet * (2*pi)

r1, r2 = [], []

rlist = np.linspace(0, 3, 100)


#Calculate2
#----------------
for i in range(len(rlist)):
    r = i * 0.03  
    
    result0 = mesolve(H_nonlin, rho0, taulist, collapse(n_th, r, theta, x, g0), [])
    for k in range(len(result0.times)):
        mat_ele0 = np.array(ptrace(result0.states[k],c1))[q1][q2]
        rho_evol0 = LA.norm(mat_ele0)
        if rho_evol0 < 0.1:
            r1.append(result0.times[k])
            break
    
    
    result1 = mesolve(H_nonlin, rho0, taulist, collapse(n_th, r, theta, y,g0), [])
    for k in range(len(result1.times)):
        mat_ele1 = np.array(ptrace(result1.states[k],c1))[q1][q2]
        rho_evol1 = LA.norm(mat_ele1)
        if rho_evol1 < 0.1:
            r2.append(result1.times[k])
            break
            
#Plotting
#--------
plt.plot(th, t, 'b');
plt.plot(th, t1, 'b--');

if q1 == 0:
    num = str(q2)
    res = num.zfill(2)
else:
    N = [q1, q2]
    res = (''.join(str(l) for l in N))
    
if c1 == 0:
    plt.title('(a)\n'+'Optical decoherence times for $P_{%s}(t)$ ' %res + ' (r = %2.1f)' %rs, fontsize=15)
if c1 == 1:
    plt.title('Mechanical decoherence times for $P_{%s}(t)$ ' %res + ' (r = %2.1f)' %rs)
    
plt.ylabel('$\kappa$ t', fontsize=15);

plt.xlabel(r'$ \theta / 2\pi $', fontsize=15);

plt.legend((strx, stry),fontsize="large",loc=2);

plt.savefig('DSME_sqDSME1.eps', format = 'eps')

plt.show()

plt.plot(rlist, r1, 'b');
plt.plot(rlist, r2, 'b--');

if q1 == 0:
    num = str(q2)
    
    res = num.zfill(2)
else:
    N = [q1, q2]
    res = (''.join(str(l) for l in N))

if c1==0:
    plt.title('(b)\n'+'Optical decoherence times for $P_{%s}(t)$ ' %res + r' ($ \theta / 2\pi $ = %2.1f)' %thet, fontsize=15)
if c1==1:
    plt.title('Mechanical decoherence times for $P_{%s}(t)$ ' %res + r' ($ \theta / 2\pi $ = %2.1f)' %thet)

plt.ylabel('$\kappa$ t', fontsize=15);

plt.xlabel('r', fontsize=15);

plt.legend((strx, stry),fontsize="large",loc=4);

plt.savefig('DSME_sqDSME2.eps', format = 'eps')

plt.show()


