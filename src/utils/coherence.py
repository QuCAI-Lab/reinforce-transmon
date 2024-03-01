'''Find T1 and T2 coherence times for a transmon qubit'''

import matplotlib.pyplot as plt
import scqubits as scq
import numpy as np
scq.settings.T1_DEFAULT_WARNING = False

temp = 0.045
tmon_charge = scq.Transmon(EJ=50, EC=1, ng=0.0, ncut=150)

newfig, newaxes = tmon_charge.plot_coherence_vs_paramvals(param_name='ng', 
                                            param_vals=np.linspace(-0.5, 0.5, 100), 
                                            noise_channels=[
                                                't1_effective',
                                                't2_effective',
                                                't1_charge_impedance',
                                                ('t1_capacitive', dict(T=temp, i=0, j=1))
                                                ],
                                                color='brown')

tmon_charge.plot_coherence_vs_paramvals(param_name='ng',  
                                     param_vals=np.linspace(-0.5, 0.5, 100), 
                                     noise_channels=[
                                     't1_charge_impedance',
                                     ('t1_capacitive', dict(T=temp, i=0, j=1, total=False))
                                     ],  
                                     color='brown');


t1_capacitive = tmon_charge.t1_effective(noise_channels=['t1_capacitive'], common_noise_options=dict(T=temp, i=0, j=1, total=False))
print(f'T1 effec. charge regime: {t1_capacitive}')
plt.show()

