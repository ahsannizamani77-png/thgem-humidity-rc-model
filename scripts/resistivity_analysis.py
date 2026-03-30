import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RH_percent = np.array([20, 40, 60, 80, 95])
rho_data = np.array([1e13, 5e11, 2e10, 1e9, 5e8])
RH_frac = RH_percent / 100.0
log_rho = np.log10(rho_data)

def fit_func(RH, A, B):
    return A - B * RH

popt, pcov = curve_fit(fit_func, RH_frac, log_rho)
A, B = popt
A_err, B_err = np.sqrt(np.diag(pcov))
print(f"Fit: log10(rho) = ({A:.2f} ± {A_err:.2f}) - ({B:.2f} ± {B_err:.2f}) * RH")

RH_smooth = np.linspace(0, 1, 100)
log_fit = fit_func(RH_smooth, A, B)
log_fit_upper = fit_func(RH_smooth, A + A_err, B - B_err)
log_fit_lower = fit_func(RH_smooth, A - A_err, B + B_err)

plt.figure(figsize=(8,5))
plt.scatter(RH_frac, log_rho, color='red', label='ESA data')
plt.plot(RH_smooth, log_fit, 'b-', label='Fit')
plt.fill_between(RH_smooth, log_fit_lower, log_fit_upper, alpha=0.3, label='1σ uncertainty')
plt.xlabel('Relative humidity (fraction)')
plt.ylabel('log10(surface resistivity) [Ω/sq]')
plt.title('FR4 Surface Resistivity vs. Humidity')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/resistivity_fit_with_errors.png', dpi=150)
plt.show()

np.savetxt('analysis/resistivity_fit_params.csv', [A, B, A_err, B_err], delimiter=',')
