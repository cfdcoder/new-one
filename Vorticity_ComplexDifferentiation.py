import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
#vortex veloctiy field : veloctiy potential = gamma/ 2*pi*R
#f(x + ih) = f(x) + ih f'(x) - h2f''(x)
#Im(f(x+ih)) = hf'(x) +0(h3)
# f'(x) = Im(f(x+ih))/h

#u',v' = gamma/ 2piR * exp(1 -(r/R)**2/2) *(y0-y, x-x0)
def velocity_perturbations(x,y, gamma, x0, y0, R):
    dx= x-x0
    dy= y-y0
    r_sq = dx**2 + dy**2
    R_sq = R**2

    exp_term= np.exp(-r_sq/R_sq/2)

    factor = gamma/ (2*np.pi*R_sq)
    u_prime = -factor * dy *exp_term
    v_prime = factor *dx* exp_term

    return u_prime, v_prime

def vorticity_complex_step(x,y,gamma, x0, y0, R, epsilon= 1e-30):
    """
    Vorticity is omega = dv/dx - du/dy
    dv/dx = Im[v(x+i*eps, y)]/eps
    du/dy = Im[u(x , y+i*eps)]/eps 
    eps: complex step size

    """
    x_complex = x.astype(complex)
    y_complex = y.astype(complex)

    #compute dv/dx
    x_pertubed = x_complex + 1j*epsilon
    _, v_at_x_plus_ih = velocity_perturbations(x_pertubed, y_complex, gamma, x0, y0, R)
    dv_dx= np.imag(v_at_x_plus_ih)/epsilon

    y_pertubed = y_complex + 1j*epsilon
    _, u_at_y_plus_ih = velocity_perturbations(x_complex, y_pertubed, gamma, x0, y0, R)
    du_dy= np.imag(u_at_y_plus_ih)/epsilon

    vorticity = dv_dx - du_dy
    return np.real(vorticity)

def vorticity_finite_difference(x,y,gamma, x0,y0, r, h=1e-6):
    """
    dv/dx= v(x+h) - v(x-h)/2h
    du/dy = u(y+h) - u(y-h)/2h

    """
    _, v_plus = velocity_perturbations(x+h, y, gamma, x0, y0, R)
    _, v_minus = velocity_perturbations(x-h, y, gamma, x0, y0, R)
    dv_dx = (v_plus - v_minus) /(2*h)

    _, u_plus = velocity_perturbations(x, y+h, gamma, x0, y0, R)
    _, u_minus = velocity_perturbations(x, y-h, gamma, x0, y0, R)
    du_dy = (u_plus - u_minus) /(2*h)

    vorticity = dv_dx - du_dy

    return vorticity


print('='*60)
print('Complex step differentialtion: Isentropic Vortex')
print('=' * 60)

gamma = 1.0
x0, y0 = 0.0, 0.0
R =0.1
grid_size = 100

x_1d =  np.linspace(-0.5,0.5,grid_size)
y_1d = np.linspace(-0.5, 0.5, grid_size)
X,Y = np.meshgrid(x_1d, y_1d)

U, V = velocity_perturbations(X,Y, gamma, x0, y0, R)
velocity_magnitude = np.sqrt(U**2 + V**2)

plt.figure(figsize=(12,8))
plt.contourf(X,Y, velocity_magnitude, levels= 20, cmap='viridis')
plt.colorbar(label='V_magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Isentropic Vortex')
plt.axis('equal')
plt.tight_layout()
plt.show

#vorticity plot

vorticity_cs = vorticity_complex_step(X,Y,gamma, x0, y0, R, epsilon= 1e-30)
plt.figure(figsize=(12,8))
plt.contourf(X,Y, vorticity_cs, levels= 20, cmap='RdBu_r')
plt.colorbar(label='Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vorticity')
plt.axis('equal')
plt.tight_layout()
plt.show

#Compare Complex and finite difference

test_x, test_y = 0.05, 0.05

for eps in [1e-6, 1e-10, 1e-20, 1e-30, 1e-50, 1e-100]:
    vort = vorticity_complex_step(np.array([test_x]), np.array([test_y]), gamma, x0, y0, R, epsilon=eps)[0]
    print(f"{eps:<15.0e} {vort:.10f}")

for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    vort = vorticity_finite_difference(np.array([test_x]), np.array([test_y]), gamma, x0, y0, R, h=h)[0]
    print(f"{h:<15.0e} {vort:.10f}")


## Vorticity analytical ###

def vorticity_analytical(x,y,gamma, x0,y0,R):
    dx= x-x0
    dy=y-y0
    r_sq =dx**2 +dy**2
    R_sq = R**2

    term1 = gamma / (2*np.pi*R_sq)
    term2 = 2 -(r_sq**2 / R_sq)
    term3 = np.exp(1 -(r_sq**2 / R_sq) / 2)

    vorticity_ana = term1*term2*term3
    return vorticity_ana

vorticity_analytical = vorticity_analytical(X,Y, gamma, x0, y0, R)
error = np.abs(vorticity_cs - vorticity_analytical)
print(f"error:{np.max(error):.2e}")
