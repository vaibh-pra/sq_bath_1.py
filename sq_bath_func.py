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

d = tensor(destroy(Nc), qeye(Nm))
c = tensor(qeye(Nc), destroy(Nm))
num_d = d.dag()*d
num_c = c.dag()*c

def sol(gamma, kappa, delta, g0, E):
    x, y = sy.symbols("x, y")
    
    C = 1j*g0**2/((gamma/2)**2 + 1)*1/(1j*delta-(kappa/2))
    D = np.conj(C)
    A = 1j*E*1/(1j*delta-(kappa/2))
    B = np.conj(A)
    values = sy.nsolve((x + C*y*x**2 - A, y + D*x*y**2 - B), (x, y), (0, 0))
    alpha = values[0]
    alpha_s = values[1] 

    X = round(LA.norm(complex(alpha)), 8)
    b = round(g0*X**2/((gamma/2)**2 + 1), 8)
    
    return [alpha, alpha_s, b]
    


def initst(Nc, Nm, p1, p2, p3, c1):

    T = 0.25 * 2 * pi
    phi = 0.0 * 2 * pi
    if c1==0:
        psi0=tensor(cos(T/2)*fock(Nc,p1)+(np.exp(1j*phi))*sin(T/2)*fock(Nc,p2), fock(Nm, p3))
        rho0=psi0*psi0.dag()
        return rho0
    if c1==1:
        psi0=tensor(fock(Nc, p3), cos(T/2)*fock(Nm,p1)+(np.exp(1j*phi))*sin(T/2)*fock(Nm,p2))
        rho0=psi0*psi0.dag()
        return rho0
    else:
        print('error')
        


def collapse(n_th, r, theta, z, g0):
#    theta = thet * 2*pi
    N = sinh(r)**2
    Neff = n_th*(cosh(r)**2 + sinh(r)**2) + sinh(r)**2
    Meff = cosh(r)*sinh(r)*(2*n_th + 1)*np.exp(1j*theta)
    M = cosh(r)*sinh(r)*np.exp(1j*theta)
    
    if z == 1:
        # Collapse operators SME
        cc0 = np.sqrt(kappa)*d
        cm0 = np.sqrt(gamma*(1.0 + n_th))*c
        cp0 = np.sqrt(gamma*(n_th))*c.dag()
        c_ops0 = [cc0,cm0,cp0]
        return c_ops0
    
    if z == 11:
        #Squeezed collapse op SME
        o0 = np.sqrt(kappa*(N + 1))*d
        o1 = np.sqrt(kappa*N)*d.dag()
        o2 = np.sqrt(kappa*M)*d
        o3 = np.sqrt(kappa*np.conj(M))*d.dag()
        op = [o0, o1]
        op1 = [o2, o3]
        SLo0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in op])
        SLo1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in op1])
        SLo = SLo0 + SLo1
        m0 = np.sqrt(gamma*(Neff + 1))*c
        m1 = np.sqrt(gamma*Neff)*c.dag()
        m2 = np.sqrt(gamma*Meff)*c
        m3 = np.sqrt(gamma*np.conj(Meff))*c.dag()
        mp = [m0, m1]
        mp1 = [m2, m3]
        SLm0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in mp])
        SLm1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in mp1])
        SLm = SLm0 + SLm1
        return [SLo, SLm]
    
    if z == 2:
        # dephsed Collapse operators_dphasd-SME
        #-------------------
        cc = np.sqrt(kappa)*d
        cm = np.sqrt(gamma*(1.0 + n_th))*(c - g0*(d.dag()*d))
        cp = np.sqrt(gamma*n_th)*(c.dag() - g0*(d.dag()*d))
        cncm = np.sqrt(gamma*((2*n_th) + 1.0))*(g0*d.dag()*d)
        c_ops = [cc,cm,cp,cncm]
        return c_ops
    
    if z == 21:
        # dephased squeezed Collapse operators dphasd-SSME
        #-------------------
        o0 = np.sqrt(kappa*(N + 1))*d
        o1 = np.sqrt(kappa*N)*d.dag()
        o2 = np.sqrt(kappa*M)*d
        o3 = np.sqrt(kappa*np.conj(M))*d.dag()
        op = [o0, o1]
        op1 = [o2, o3]
        SLo0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in op])
        SLo1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in op1])
        SLo = SLo0 + SLo1
        Dm0 = np.sqrt(gamma*(Neff + 1))*(c - g0*(num_d))
        Dm1 = np.sqrt(gamma*Neff)*(c.dag() - g0*(num_d))
        Dm2 = np.sqrt(gamma*Meff)*(c - g0*(num_d))
        Dm3 = np.sqrt(gamma*np.conj(Meff))*(c.dag() - g0*(num_d))
        Dm4 = np.sqrt(4*gamma*n_th*(cosh(r)**2 + sinh(r)**2 + 2*cosh(r)*sinh(r)*cos(theta)))*(g0*d.dag()*d)
        Dmp = [Dm0, Dm1]
        Dmp1 = [Dm2, Dm3]
        Dmp2 = [Dm4]
        DSLm0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in Dmp])
        DSLm1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in Dmp1])
        DSLm2 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in Dmp2])
        DSLm = DSLm0 + DSLm1
        dDm0 = np.sqrt(gamma*(2*Neff + 1))*(g0*d.dag()*d)
        dDm1 = (g0*d.dag()*d)
        D = 2*gamma*cosh(r)*sinh(r)*cos(theta)*(2*n_th + 1)
        dDmp = [dDm0]
        dDmp1 = [dDm1]
        dDSLm0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in dDmp])
        dDSLm1 = sum([D*spre(a)*spost(a) - 0.5 * D*spre(a*a) - 0.5 * D*spost(a*a) for a in dDmp1])
        dDSLm = DSLm + (dDSLm0 + dDSLm1)
        return [SLo, dDSLm]
        
    
    if z == 3:
        # dressed state Collapse operators_DSME
        #-------------------
        cc1 = np.sqrt(kappa)*d
        cm1 = np.sqrt(gamma*(1.0 + n_th))*(c - g0*(d.dag()*d))
        cp1 = np.sqrt(gamma*n_th)*(c.dag() - g0*(d.dag()*d))
        cp11 = 2*np.sqrt(gamma*n_th)*(g0*d.dag()*d)
        c_ops1 = [cc1,cm1,cp1,cp11]
        return c_ops1
    
    if z == 31:
        # dressed state squeezed Collapse operators_SDSME
        #-------------------
        o0 = np.sqrt(kappa*(N + 1))*d
        o1 = np.sqrt(kappa*N)*d.dag()
        o2 = np.sqrt(kappa*M)*d
        o3 = np.sqrt(kappa*np.conj(M))*d.dag()
        op = [o0, o1]
        op1 = [o2, o3]
        SLo0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in op])
        SLo1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in op1])
        SLo = SLo0 + SLo1
        Dm0 = np.sqrt(gamma*(Neff + 1))*(c - g0*(num_d))
        Dm1 = np.sqrt(gamma*Neff)*(c.dag() - g0*(num_d))
        Dm2 = np.sqrt(gamma*np.conj(Meff))*(c - g0*(num_d))
        Dm3 = np.sqrt(gamma*Meff)*(c.dag() - g0*(num_d))
        Dm4 = np.sqrt(4*gamma*n_th*(cosh(r)**2 + sinh(r)**2 + 2*cosh(r)*sinh(r)*cos(theta)))*(g0*d.dag()*d)
        Dmp = [Dm0, Dm1]
        Dmp1 = [Dm2, Dm3]
        Dmp2 = [Dm4]
        DSLm0 = sum([spre(a)*spost(a.dag()) - 0.5 * spre(a.dag()*a) - 0.5 * spost(a.dag()*a) for a in Dmp])
        DSLm1 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in Dmp1])
        DSLm2 = sum([spre(a)*spost(a) - 0.5 * spre(a*a) - 0.5 * spost(a*a) for a in Dmp2])
        DSLm = DSLm0 + DSLm1 + DSLm2
        return [SLo, DSLm]
    
    else:
        print('Input/s not in list')    
        return 0
        


def legstr(x):
    if x == 1:
        str1 = 'SME'
        return str1
    elif x == 2:
        str1 = 'dephased-SME'
        return str1
    elif x == 3:
        str1 = 'thermal DSME'
        return str1
    elif x == 11:
        str1 = 'squeezed SME'
        return str1
    elif x == 21:
        str1 = 'squeezed dephased-SME'
        return str1
    elif x == 31:
        str1 = 'squeezed thermal DSME'
        return str1
    else:
        print('Input/s not in list')
        return 0        
