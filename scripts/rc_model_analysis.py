import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import epsilon_0

params = np.loadtxt('analysis/resistivity_fit_params.csv', delimiter=',')
A, B, A_err, B_err = params

def resistivity(RH):
    return 10 ** (A - B * RH)

def resistivity_upper(RH):
    return 10 ** ((A + A_err) - (B - B_err) * RH)

def resistivity_lower(RH):
    return 10 ** ((A - A_err) - (B + B_err) * RH)

hole_radius = 0.015
hole_length = 0.04
pitch = 0.07
epsilon_r = 4.5

r_outer = pitch / 2
r_inner = hole_radius
C = 2 * np.pi * epsilon_0 * epsilon_r * hole_length / np.log(r_outer / r_inner)
print(f"Capacitance per hole: {C*1e12:.2f} pF")

RH_range = np.linspace(0.2, 0.95, 50)
rho_s = resistivity(RH_range)
rho_s_upper = resistivity_upper(RH_range)
rho_s_lower = resistivity_lower(RH_range)

tau = rho_s * C
tau_upper = rho_s_upper * C
tau_lower = rho_s_lower * C

RH_plot = [0.2, 0.6, 0.95]
initial_gain = 1000
final_gain = 500
dt = 0.01
t_max = 10
time = np.arange(0, t_max, dt)

plt.figure(figsize=(10,6))
for rh in RH_plot:
    tau_val = resistivity(rh) * C
    gain = final_gain + (initial_gain - final_gain) * np.exp(-time / tau_val)
    plt.plot(time, gain, label=f'{rh*100:.0f}% RH, τ = {tau_val:.2f} s')
plt.xlabel('Time (s)')
plt.ylabel('Gain')
plt.title('THGEM Gain Evolution with Humidity (RC Model)')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/gain_evolution_rc.png', dpi=150)
plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(RH_range*100, tau, 'b-', label='Nominal')
plt.fill_between(RH_range*100, tau_lower, tau_upper, alpha=0.3, label='1σ uncertainty')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Charging‑up Time Constant (s)')
plt.title('Humidity Effect on Charging‑up Time Constant')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/tau_vs_humidity_rc.png', dpi=150)
plt.show()

df = pd.DataFrame({
    'RH_percent': RH_range*100,
    'rho_s_Ohm_sq': rho_s,
    'tau_charging_s': tau,
    'tau_lower_s': tau_lower,
    'tau_upper_s': tau_upper
})
df.to_csv('results/processed/rc_model_summary.csv', index=False)
print("Results saved to results/processed/rc_model_summary.csv")
