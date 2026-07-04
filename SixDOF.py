import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Cessna172_6DOF:
    def __init__(self):
        # Cessna 172 Mass and Inertia
        self.m = 1114.0       # kg
        self.Ixx = 1285.0
        self.Iyy = 1825.0
        self.Izz = 2667.0
        self.Ixz = 0.0
        
        # Geometry and Environment
        self.S = 16.2         # Wing area (m^2)
        self.c = 1.49         # Mean aerodynamic chord (m)
        self.rho = 1.225      # Air density at sea level (kg/m^3)
        self.g = 9.81
        
        # Simplified Aerodynamic Coefficients
        self.CL0 = 0.31
        self.CLa = 5.143      # Lift curve slope
        self.CLde = 0.438     # Elevator effect on lift
        
        self.CD0 = 0.031
        self.K = 0.054        # Induced drag factor
        
        self.Cm0 = -0.015
        self.Cma = -0.89      # Pitch stability
        self.Cmq = -12.4      # Pitch damping
        self.Cmde = -1.28     # Elevator effect on pitch
        
        # Engine Thrust (Assuming constant thrust to match cruise drag)
        self.thrust = 1500.0  # Newtons

    def calculate_forces_and_moments(self, t, state):
        u, v, w, p, q, r, phi, theta, psi, x, y, z = state
        
        # --- Control Input (The Pull-Up Maneuver) ---
        # Deflect elevator upward by 0.1 radians after 2 seconds
        if t > 2.0:
            delta_e = -0.1 
        else:
            delta_e = 0.0
            
        # --- Air data ---
        V = np.sqrt(u**2 + w**2)
        if V == 0: V = 0.001 # Prevent division by zero
        q_bar = 0.5 * self.rho * V**2
        alpha = np.arctan2(w, u)
        
        # --- Aerodynamic Coefficients ---
        CL = self.CL0 + self.CLa * alpha + self.CLde * delta_e
        CD = self.CD0 + self.K * CL**2
        Cm = self.Cm0 + self.Cma * alpha + self.Cmq * (q * self.c / (2 * V)) + self.Cmde * delta_e
        
        # --- Aerodynamic Forces and Moments ---
        L_aero = q_bar * self.S * CL
        D_aero = q_bar * self.S * CD
        
        M_aero = q_bar * self.S * self.c * Cm
        
        # Rotate Lift and Drag into Body Frame X and Z
        Fx_aero = L_aero * np.sin(alpha) - D_aero * np.cos(alpha)
        Fz_aero = -L_aero * np.cos(alpha) - D_aero * np.sin(alpha)
        
        # --- Gravity Forces ---
        Fx_grav = -self.m * self.g * np.sin(theta)
        Fy_grav = self.m * self.g * np.sin(phi) * np.cos(theta)
        Fz_grav = self.m * self.g * np.cos(phi) * np.cos(theta)
        
        # --- Total Forces and Moments ---
        Fx = Fx_grav + Fx_aero + self.thrust
        Fy = Fy_grav + 0.0 
        Fz = Fz_grav + Fz_aero
        
        L = 0.0 # No roll applied
        M = M_aero
        N = 0.0 # No yaw applied
        
        return Fx, Fy, Fz, L, M, N

    def equations_of_motion(self, t, state):
        u, v, w, p, q, r, phi, theta, psi, x, y, z = state
        Fx, Fy, Fz, L, M, N = self.calculate_forces_and_moments(t, state)
        
        # 1. Force Equations
        u_dot = (Fx / self.m) - q*w + r*v
        v_dot = (Fy / self.m) - r*u + p*w
        w_dot = (Fz / self.m) - p*v + q*u
        
        # 2. Moment Equations 
        gamma = self.Ixx * self.Izz - self.Ixz**2
        p_dot = (self.Izz*L + self.Ixz*N - self.Ixz*(self.Iyy - self.Ixx - self.Izz)*p*q + (self.Ixx*self.Izz - self.Izz**2 - self.Ixz**2)*q*r) / gamma
        q_dot = (M - (self.Ixx - self.Izz)*p*r - self.Ixz*(p**2 - r**2)) / self.Iyy
        r_dot = (self.Ixz*L + self.Ixx*N + self.Ixz*(self.Iyy - self.Ixx - self.Izz)*q*r + (self.Ixx**2 - self.Ixx*self.Iyy + self.Ixz**2)*p*q) / gamma
        
        # 3. Kinematic Equations
        phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)
        
        # 4. Navigation Equations
        x_dot = u * np.cos(theta) * np.cos(psi) + v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        y_dot = u * np.cos(theta) * np.sin(psi) + v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
        z_dot = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(phi) * np.cos(theta)
        
        return [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot, x_dot, y_dot, z_dot]

# --- Simulation Setup ---
sim = Cessna172_6DOF()

# Initial State: Flying level at ~50 m/s (approx 100 knots)
# [u, v, w, p, q, r, phi, theta, psi, x, y, z]
initial_state = [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1000.0]

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

print("Simulating Cessna pull-up maneuver...")
sol = solve_ivp(sim.equations_of_motion, t_span, initial_state, t_eval=t_eval, method='RK45')

# --- Plotting the Results ---
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot Altitude
axs[0].plot(sol.t, -sol.y[11], color='blue', linewidth=2)
axs[0].set_title('Altitude (m) - Notice the climb after t=2s')
axs[0].grid(True)

# Plot Pitch Angle (Theta)
axs[1].plot(sol.t, np.degrees(sol.y[7]), color='orange', linewidth=2)
axs[1].set_title('Pitch Angle (Degrees)')
axs[1].grid(True)

# Plot Angle of Attack (Alpha)
# Recalculate alpha for plotting: alpha = arctan(w/u)
alpha_history = np.degrees(np.arctan2(sol.y[2], sol.y[0]))
axs[2].plot(sol.t, alpha_history, color='green', linewidth=2)
axs[2].set_title('Angle of Attack (Degrees)')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)

plt.tight_layout()
plt.show()-