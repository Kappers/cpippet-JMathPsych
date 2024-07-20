'''
PIPPET: Phase Inference from Point Process Event Timing [1]
    + mPIPPET: multiple event streams [1]
    + pPIPPET: pattern inference [2]
    + ctPIPPET: oscillatory PIPPET

Python variant of Jonathan Cannon's original MATLAB implementation:
    https://github.com/joncannon/PIPPET

[1] Expectancy-based rhythmic entrainment as continuous Bayesian inference.
    Cannon J (2021)  PLOS Computational Biology 17(6): e1009025.
[2] Modeling enculturated bias in entrainment to rhythmic patterns.
    Kaplan T, Cannon J, Jamone L & Pearce M (2021) - In Review.

@Tom Kaplan: t.m.kaplan@qmul.ac.uk
'''
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from scipy.optimize import fsolve
from scipy.stats import norm
import PIPPET

TWO_PI = 2 * np.pi

class cPIPPET(PIPPET):
    ''' Oscillatory (wrapped) PIPPET with movement and tapping '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.z_s = np.ones(self.n_ts, dtype=np.clongdouble)
        self.z_s[0] = np.exp(complex(-self.params.V_0/2, self.params.mu_0))

    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float) -> complex:
        ''' Posterior update for a time step '''

        dz_sum = 0
        for s_i in range(self.n_streams):
            blambda = self.streams[s_i].zlambda(mu_prev, V_prev, self.params.tau)
            z_hat = self.streams[s_i].z_hat(mu_prev, V_prev, blambda, self.params.tau)
            dz = blambda*(z_hat-z_prev)*self.params.dt
            dz_sum += dz

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_perp = self.params.tau * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par) - dz_sum

        #z = z_prev + z_prev*complex(-(self.params.sigma_phi**2)/2, self.params.tau)*self.params.dt - dz_sum
        z_norm = abs(z)
        if z_norm>1:
            z = z/z_norm * 0.9999
            print('o dang znorm '+str(z_norm))
        mu, V_s = PIPPETStream.z_mu_V(z)

        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                z = self.streams[s_i].z_hat(mu, V_s, self.streams[s_i].zlambda(mu, V_s, self.params.tau), self.params.tau)
                z_norm = abs(z)
                if z_norm>1:
                    z = z/z_norm * 0.9999
                    print('ono znorm '+str(z_norm))
                    print('Vprev ' +str(V_prev))
                    print('muprev ' +str(mu_prev))
                    print('zlam '+str(self.streams[s_i].zlambda(mu, V_s, self.params.tau)))
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)

                self.grad[t_i] =  -np.log(self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] +=  np.log(self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)
                self.grad[t_i] =  -np.log(1-self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] +=  np.log(1-self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] /= .02


        return z

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            z = self.step(i, z_prev, mu_prev, V_prev)
            mu, V = PIPPETStream.z_mu_V(z)
            # Noise
            mu += np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()
            V *= np.exp(np.sqrt(self.params.dt) * self.params.eta_V * np.random.randn())
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = np.exp(complex(-V/2, mu))

