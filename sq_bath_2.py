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



tlist = np.linspace(0, 50, 600)
taulist = tlist * kappa
delta = 0.0
t = 0.25 * 2 * pi
phi = 0.0 * pi
d = tensor(destroy(Nc), qeye(Nm))
c = tensor(qeye(Nc), destroy(Nm))
num_d = d.dag()*d
num_c = c.dag()*c

# Initial fock state
#-------------------
#psi0=tensor(fock(4,0), cos(t/2)*fock(5,0)+(np.exp(1j*phi))*sin(t/2)*fock(5,3) )
psi0=tensor(cos(t/2)*fock(2,0)+(np.exp(1j*phi))*sin(t/2)*fock(2,1), fock(3,0))
#psi0=tensor(coherent(4,sqrt(0))+coherent(4,sqrt(1)), coherent(5,sqrt(3)))/np.sqrt(2)
rho0=psi0*psi0.dag()
#rho0_fock11 = ptrace(rho0,0)
#rho0_fock12 = ptrace(rho0,1)

# Standard master equation solver
#--------------------------------
#L = sum([spre(c)*spost(c.dag()) - 0.5 * spre(c.dag()*c) - 0.5 * spost(c.dag()*c) for c in c_ops])
#H_nonlin = -delta * (num_d) + num_c - g0 * (c + c.dag()) * num_d - 1j*E*(c.dag()-c)

H_nonlin1 = 10**3 * (num_d) + num_c - (0.01 * num_d)**2
H_nonlin2 = 10**3 * (num_d) + num_c - (0.01 * num_d)**2
H_nonlin3 = 10**3 * (num_d) + num_c - (0.01 * num_d)**2
H_nonlin4 = 10**3 * (num_d) + num_c - (0.01* num_d)**2

#alpha = sol(gamma, kappa, delta, g0, E)[0]
#alpha_s = sol(gamma, kappa, delta, g0, E)[1]
#b = sol(gamma, kappa, delta, g0, E)[2]

#deltaP = delta - g0 * b

#H_nonlinear = -deltaP*num_d+num_c+g0*(complex(alpha_s)*d+complex(alpha)*d.dag())*(c+c.dag())+g0*num_d*(c+c.dag())

thet = 0.0
r = 0.0
theta = thet*2*pi

result0 = mesolve(H_nonlin1, psi0, taulist, collapse(0,r,theta,2,0.8), [])
result1 = mesolve(H_nonlin2, psi0, taulist, collapse(0,r,theta,2,0.8), [])


result2 = mesolve(H_nonlin3, psi0, taulist, collapse(20,r,theta,2,0.8), [])
result3 = mesolve(H_nonlin4, psi0, taulist, collapse(20,r,theta,2,0.8), [])

#result = mesolve(H_non_linear, rho0, taulist, c_ops, [])
#expt = mesolve(H_non_linear, psi0, taulist, c_ops, [d.dag()*d , c.dag()*c])

rho_evol0, rho_evol1, rho_evol2, rho_evol3 = [], [], [], []
for i in range(len(result0.times)):
    mat_ele0 = np.array(ptrace(result0.states[i],0))[0][1]
    rho_evol0.append(LA.norm(mat_ele0))
for i in range(len(result1.times)):
    mat_ele1 = np.array(ptrace(result1.states[i],0))[0][1]
    rho_evol1.append(LA.norm(mat_ele1))
for i in range(len(result2.times)):
    mat_ele2 = np.array(ptrace(result2.states[i],0))[0][1]
    rho_evol2.append(LA.norm(mat_ele2))
for i in range(len(result3.times)):
    mat_ele3 = np.array(ptrace(result3.states[i],0))[0][1]
    rho_evol3.append(LA.norm(mat_ele3))
#for i in range (len(result.times)):
   # mat_ele = np.array(ptrace(result.states[i],0))[0][1]
   # rho_evol.append(LA.norm(mat_ele))


#rho_ss0 = steadystate(H_nonlin, sc_ops0)
#rho_ss1 = steadystate(H_nonlin, sc_ops1)

plt.plot(result0.times, rho_evol0,'r',lw=1.2)
plt.plot(result1.times, rho_evol1,'r--',lw=1.5)
plt.plot(result2.times, rho_evol2,'b-.',lw=1.5)
plt.plot(result3.times, rho_evol3,'b:',lw=1.8)
#plt.plot(result.times, rho_evol);
#plt.plot(result.times, expt1.expect[0]);

plt.xlabel('$\kappa$ t', fontsize=15);

plt.ylabel(r'$ P_{03} (t) $', fontsize=15);

plt.title('(a)\n'+'Squeezed reservoir' + '  (r = %2.1f,  '%r + r'$ \theta / 2\pi $ = %2.1f).' %thet, fontsize=15)

l = plt.legend(["Dephased-SME($n_{th}=0$)", "DSME($n_{th}=0$)", "Dephased-SME($n_{th}=20$)", "DSME($n_{th}=20$)"]
               , fontsize="small", loc=4)
#l = plt.legend(['$g_0=0.01 \omega_m$', '$g_0 = 0.1 \omega_m$', '$g_0 = 0.5 \omega_m$', '$g_0 = 1 \omega_m$']
#               , fontsize="large", loc=1)
for text in l.get_texts():
    text.set_color("black")
    
plt.savefig('prev.eps', format='eps')

plt.show()
