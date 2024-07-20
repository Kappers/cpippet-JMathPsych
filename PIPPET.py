'''
PIPPET: Phase Inference from Point Process Event Timing [1]
    + mPIPPET: multiple event streams [1]
    + pPIPPET: pattern inference [2]
    + cPIPPET: oscillatory PIPPET
    + gcPATIPPET: gradient oscillatory pippet with tempo inference
    + vcPATIPPET: variational oscillatory pippet with tempo inference

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
import warnings
warnings.filterwarnings("error")

TWO_PI = 2 * np.pi

@dataclass(init=True, repr=True)
class TemplateParams:
    ''' Expectation template and respective (stimulus) event timing '''
    e_times: np.ndarray   # Observed event times
    e_means: np.ndarray   # Expected event times (mean phase)
    e_vars: np.ndarray    # Variance of expected event times
    e_lambdas: np.ndarray # Strength of expected event times
    lambda_0: float       # Strength of uniform expectation (ignorability)
    label: str            # Identifier/label for analysis


@dataclass(init=True, repr=True)
class PIPPETParams:
    ''' Configuration for PIPPET model - parameters and expectation templates '''
    templates: list = field(default_factory=list)

    mu_0: float = 0.0         # Initial estimated phase
    V_0: float = 0.0002       # Initial variance
    sigma_phi: float = 0.05   # Generative model phase noise
    sigma_omega: float = 0.05 # Generative model tempo noise (for ctPIPPET)
    sigma_omega_global: float = 0.0 # Generative model diffusion to other oscillators, random phase
        
    movement_precision: float = 0.0 # Continuous influence of movement on internal time
    movement_updating:  float = 0.0 # Continuous influence of internal time on movement
    start_tapping: float = 0.1      # Time to start tapping.

    dwell_stability: float = 1.0    # Attractiveness of finger dwell position
    tap_stability: float = 10.0     # Maximum attractiveness of tap goal point
    tap_target: float = -.5   # tap goal point

    eta_mu: float = 0.0       # Internal phase noise
    eta_V: float = 0.0        # Internal variance noise
    eta_e: float = 0.0        # Internal event noise
    eta_alpha: float = 0.0        # Motor noise
    eta_e_share: bool = False # Shared event noise across templates (for pPIPPET, set True)
    
        
    dt: float = 0.001         # Integration time step
    overtime: float = 0.0     # Time buffer for simulation
    t0: float = 0.0           # Starting time for simulation with respect to event times
    tmax: float = np.nan      # Maximum time for simulation (otherwise based on event times)

    omega: float = 1.0       # Tempo-like dial for cPIPPET
    omega_p: float = 1.0   # Preferred tempo in ctPIPPET
    omega_p_tendency: float = 1.0 # Tendency towards preferred tempo
    omega_c_coef: float = 1.0 # Coupling coefficient
    
    tempo_scaling: bool = False #scale v_i with tempos
    m_span: int = 40          # terms included in approximation to infinite sum
    continuous_expectation: bool = True #include continuous effect of expectations
    
    
    verbose: bool = False # debugging output

    def add(self, times: np.ndarray, means: np.ndarray, vars_: np.ndarray, lambdas: np.ndarray, lambda_0: float, label: str):
        ''' Add an expectation template, which corresponds to either:
            (1) a unique event stream for mPIPPET,
            (2) a separate expectation template for pPIPPET
        '''
        self.templates.append(TemplateParams(times, means, vars_, lambdas, lambda_0, label))

class PIPPETStream:
    ''' Variational filtering equations for PIPPET, see Methods of [1] or [2] '''

    def __init__(self, params: TemplateParams, tempo_scaling: bool, m_span: int):
        self.params = params
        self.lambda_0 = params.lambda_0
        self.e_times_p = params.e_times
        self.tempo_scaling = tempo_scaling
        # For cPIPPET:
        self.M = np.arange(-m_span, m_span+1, 1)
        self.cs = np.empty((self.params.e_means.size, self.M.size), dtype=np.clongdouble)
        self.e_means_ = self.params.e_means.reshape(-1, 1)
        self.e_vars_ = self.params.e_vars.reshape(-1, 1)

    def mu_i(self, mu: float, V: float) -> float:
        return (mu/V + self.params.e_means/self.params.e_vars)/(1/V + 1/self.params.e_vars)

    def K_i(self, V: float) -> float:
        return 1/(1/V + 1/self.params.e_vars)

    def lambda_i(self, mu: float, V: float) -> float:
        gauss = norm.pdf(mu, loc=self.params.e_means, scale=(self.params.e_vars + V)**0.5)
        return self.params.e_lambdas * gauss

    def lambda_hat(self, mu: float, V:float) -> float:
        return self.lambda_0 + np.sum(self.lambda_i(mu, V))

    def mu_hat(self, mu: float, V: float) -> float:
        mu_hat = self.lambda_0 * mu
        mu_hat += np.sum(self.lambda_i(mu, V) * self.mu_i(mu, V))
        return mu_hat / self.lambda_hat(mu, V)

    def V_hat(self, mu_curr: float, mu_prev: float, V: float) -> float:
        V_hat = self.lambda_0 * (V + (mu_prev-mu_curr)**2)
        a = self.lambda_i(mu_prev, V)
        b = self.K_i(V) + (self.mu_i(mu_prev, V)-mu_curr)**2
        V_hat += np.sum(a * b)
        return V_hat / self.lambda_hat(mu_prev, V)

    ######### cPIPPET #######################
    ###########################################

    @staticmethod
    def z_mu_V(z: complex) -> tuple[float,float]:
        return np.angle(z), -2*np.log(np.abs(z))

    @staticmethod
    def y_mu_V(y: complex, p: float) -> tuple[float,float]:
        return np.angle(y), -2*np.log(np.abs(y/p))

    def zlambda(self, mu: float, V: float, omega: float) -> float:
        if self.tempo_scaling:
            v = self.params.e_vars * omega**2
        else:
            v = self.params.e_vars
        
        self.cs.real = -(self.M**2) * ((V+v)/2).reshape(-1, 1)
        self.cs.imag = -self.M*(mu - self.params.e_means).reshape(-1, 1)
        y = np.sum(self.params.e_lambdas*omega/TWO_PI * np.exp(self.cs).real.sum(axis=1))
        return self.lambda_0*omega/TWO_PI + y

    def z_hat(self, mu: float, V: float, blambda: float, omega: float) -> complex:
        if self.tempo_scaling:
            v = self.e_vars_ * omega**2
        else:
            v = self.e_vars_   
        self.cs.real = -(V*self.M**2)/2 - (v * (self.M + 1)**2)/2
        self.cs.imag = -self.M*(mu - self.params.e_means).reshape(-1, 1) + self.e_means_
        z_hat_i = self.params.e_lambdas*omega/TWO_PI * np.exp(self.cs).sum(axis=1)
        y = self.lambda_0*omega/TWO_PI * np.exp(complex(-V/2, mu)) + np.sum(z_hat_i)
        return 1/blambda * y

class PIPPET(ABC):
    ''' Base class for PIPPET inference problems '''

    def __init__(self, params: PIPPETParams):
        self.params = params
        # Create unique streams/patterns for (mp)PIPPET filtering, based on params
        self.streams = []
        self.labels = []
        for p in params.templates:
            self.streams.append(PIPPETStream(p, self.params.tempo_scaling, self.params.m_span))
            self.labels.append(p.label)
        self.n_streams = len(self.streams)
        self.event_n = np.zeros(self.n_streams).astype(int)

        # Pre-compute shared internal noise, if appropriate
        if params.eta_e_share:
            noise = np.random.randn(*self.streams[0].e_times_p.shape) * self.params.eta_e
            for s_i in range(self.n_streams):
                self.streams[s_i].e_times_p += noise
        else:
            for s_i in range(self.n_streams):
                noise = np.random.randn(*self.streams[s_i].e_times_p.shape) * self.params.eta_e
                self.streams[s_i].e_times_p += noise
        # Ensure events (perturbed by noise) don't occur at negative time
        for s_i in range(self.n_streams):
            self.streams[s_i].e_times_p[self.streams[s_i].e_times_p < 0] = 0.0

        # Timing of simulation
        max_times = []
        for s in self.streams:
            if len(s.e_times_p)>0:
                max_times.append(s.e_times_p[-1])
            else:
                max_times.append(0.0)
            
        self.tmax = max(max_times) + params.overtime

        self.ts = np.arange(self.params.t0, self.tmax+self.params.dt, step=self.params.dt)
        self.n_ts = self.ts.shape[0]
        # Initialise sufficient statistics
        self.mu_s = np.zeros(self.n_ts)
        self.mu_s[0] = self.params.mu_0
        self.V_s = np.zeros(self.n_ts)
        self.V_s[0] = self.params.V_0
        self.idx_event = set()
        self.event_stream = defaultdict(set)
        # Gradient of Lambda
        self.grad = np.zeros((self.n_ts, self.n_streams))
        # Surprisal
        self.surp = np.zeros((self.n_ts, self.n_streams, 2))

    def is_onset(self, t_prev: float, t: float, s_i: int, stim: bool=True) -> bool:
        ''' Check whether an event is observed on this time-step '''
        evts = self.streams[s_i].e_times_p if stim else self.streams[s_i].params.e_means
        if self.event_n[s_i] < len(evts):
            return t_prev <= evts[self.event_n[s_i]] <= t
        return False

    @abstractmethod
    def step(self) -> tuple[float, float]:
        ''' Posterior update for a time step '''
        mu, V = None, None
        return mu, V

    @abstractmethod
    def run(self) -> None:
        ''' Simulation for entire stimulus (i.e. all time steps) '''
        for i in range(1, self.n_ts):
            pass # At least, this should call self.step()


class mPIPPET(PIPPET):
    ''' PIPPET with multiple event streams '''

    def step(self, t_i: float, mu_prev: float, V_prev: float) -> tuple[float, float]:
        ''' Posterior update for a time step '''

        # Internal phase noise
        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        # Sum dmu across event streams
        dmu_sum = 0
        for s_i in range(self.n_streams):
            dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
            dmu_sum += dmu
        mu = mu_prev + self.params.dt*(1 - dmu_sum) + noise

        # Sum dV across event streams
        dV_sum = 0
        for s_i in range(self.n_streams):
            dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
            dV_sum += dV
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV_sum)

        # Update posterior based on events in any stream
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                mu_new = self.streams[s_i].mu_hat(mu, V)
                V = self.streams[s_i].V_hat(mu_new, mu, V)
                mu = mu_new
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i] =  -np.log(1-self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i] +=  np.log(1-self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i] /= .02

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            mu, V = self.step(i, mu_prev, V_prev)
            self.mu_s[i] = mu
            self.V_s[i] = V


class pPIPPET(PIPPET):
    ''' PIPPET with pattern (i.e. template) inference '''

    def __init__(self, params: PIPPETParams, prior: np.ndarray):
        super().__init__(params)

        # Track likelihoods and big Lambdas per pattern
        self.n_m = self.n_streams
        self.L_s = np.zeros(self.n_ts)
        self.L_ms = np.zeros((self.n_ts, self.n_m))
        self.p_m = np.zeros((self.n_ts, self.n_m))
        self.p_m[0] = prior
        self.p_m[0] = self.p_m[0]/self.p_m[0].sum()

        # Initialise big Lambdas using mu_0 and V_0
        for s_i, m in enumerate(self.streams):
            self.L_ms[0, s_i] = m.lambda_hat(self.mu_s[0], self.V_s[0])
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0])

    def step(self, s_i: int, mu_prev: float, V_prev: float, is_event: bool=False) -> tuple[float, float]:
        ''' Posterior step for a given pattern '''

        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
        mu = mu_prev + self.params.dt*(1 - dmu) + noise

        dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV)

        if is_event:
            mu_new = self.streams[s_i].mu_hat(mu, V)
            V = self.streams[s_i].V_hat(mu_new, mu, V)
            mu = mu_new

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, for all patterns '''

        # For each time step
        for i in range(1, self.n_ts):
            lambda_prev = self.L_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]

            mu_ms = np.zeros(self.n_m)
            V_ms = np.zeros(self.n_m)

            t_prev, t = self.ts[i-1], self.ts[i]

            # For each pattern
            for s_i in range(self.n_m):
                lambda_m_prev = self.L_ms[i-1, s_i]
                prev_p_m = self.p_m[i-1, s_i]

                # Update p_m based on event observations (or absence of them)
                is_event = self.is_onset(t_prev, t, s_i)
                d_p_m = prev_p_m * (lambda_m_prev/lambda_prev - 1)
                if not is_event:
                    d_p_m *= -self.params.dt * lambda_prev
                self.p_m[i, s_i] = prev_p_m + d_p_m

                # Update posterior and lambda_m
                mu_m, V_m = self.step(s_i, mu_prev, V_prev, is_event)
                lambda_m = self.streams[s_i].lambda_hat(mu_m, V_m)

                self.L_ms[i, s_i] = lambda_m
                mu_ms[s_i] = mu_m
                V_ms[s_i] = V_m

                if is_event:
                    self.event_n[s_i] += 1
                    self.idx_event.add(i)
                    self.event_stream[i].add(s_i)

            # Marginalize across patterns
            self.mu_s[i] = np.sum(self.p_m[i] * mu_ms)
            self.L_s[i] = np.sum(self.p_m[i] * self.L_ms[i])
            self.V_s[i] = np.sum(self.p_m[i] * V_ms)
            self.V_s[i] += np.sum(self.p_m[i]*(1 - self.p_m[i])*np.power(mu_ms, 2))
            for m in range(self.n_m):
                for n in range(self.n_m):
                    if m != n:
                        self.V_s[i] -= self.p_m[i,m]*self.p_m[i,n]*mu_ms[m]*mu_ms[n]


class cPIPPET(PIPPET):
    ''' Oscillatory (wrapped) PIPPET '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.z_s = np.ones(self.n_ts, dtype=np.clongdouble)
        self.z_s[0] = np.exp(complex(-self.params.V_0/2, self.params.mu_0))

    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float) -> complex:
        ''' Posterior update for a time step '''

        dz_sum = 0
        for s_i in range(self.n_streams):
            blambda = self.streams[s_i].zlambda(mu_prev, V_prev, self.params.omega)
            z_hat = self.streams[s_i].z_hat(mu_prev, V_prev, blambda, self.params.omega)
            if self.params.continuous_expectation:
                dz = blambda*(z_hat-z_prev)*self.params.dt
            else:
                dz = 0
            dz_sum += dz

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_perp = self.params.omega * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par) - dz_sum

        #z = z_prev + z_prev*complex(-(self.params.sigma_phi**2)/2, self.params.omega)*self.params.dt - dz_sum
        z_norm = abs(z)
        if z_norm>1:
            z = z/z_norm * 0.9999
            print('o dang znorm '+str(z_norm))
        
        mu, V_s = PIPPETStream.z_mu_V(z)
        
        # Noise
        mu += np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()
        V_s *= np.exp(np.sqrt(self.params.dt) * self.params.eta_V * np.random.randn())
        z = np.exp(complex(-V_s/2, mu))
        
        
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                
                mu, V_s = PIPPETStream.z_mu_V(z)
                z = self.streams[s_i].z_hat(mu, V_s, self.streams[s_i].zlambda(mu, V_s, self.params.omega), self.params.omega)
                z_norm = abs(z)
                #print('s = '+str(s_i)+' t = '+str(t)+ ' znorm '+str(z_norm))
                if z_norm>1:
                    z = z/z_norm * 0.9999
                    print('ono znorm '+str(z_norm))
                    print('Vprev ' +str(V_prev))
                    print('muprev ' +str(mu_prev))
                    print('zlam '+str(self.streams[s_i].zlambda(mu, V_s, self.params.omega)))
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                #self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                #self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)

                #self.grad[t_i] =  -np.log(self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.omega)*self.params.dt)
                #self.grad[t_i] +=  np.log(self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.omega)*self.params.dt)
                #self.grad[t_i] /= .02
            else:
                #self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                #self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)
                #self.grad[t_i] =  -np.log(1-self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.omega)*self.params.dt)
                #self.grad[t_i] +=  np.log(1-self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.omega)*self.params.dt)
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
 
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = z

            
            
            

class movingPIPPET(cPIPPET):
    '''PIPPET with circular movement '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.alpha_s = np.ones(self.n_ts, dtype =np.clongdouble)
        self.alpha_s[0] = self.z_s[0] #nan
        self.tapping = False
        self.params.delay_timesteps = int(self.params.delay / self.params.dt)


    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float, alpha_prev: float):
        ''' Posterior update for a time step '''
        alpha = alpha_prev * np.exp(1j * (self.params.omega )*self.params.dt)
        b = 1.3
        k = 2/b * np.tan(np.abs(z_prev)*b)
        
        if (t_i > self.params.delay_timesteps) & (np.cos(np.angle(alpha_prev)) < np.cos(self.params.sync_radius)):
            mu_delayed = self.mu_s[t_i-self.params.delay_timesteps-1]

            alpha_delayed = self.alpha_s[t_i-self.params.delay_timesteps-1]
            alpha *= np.exp(1j * self.params.dt * self.params.movement_updating * k * np.sin(mu_delayed-np.angle(alpha_delayed)))
        alpha *= np.exp(1j * np.sqrt(self.params.dt) * self.params.eta_alpha* np.random.randn())
        
        if self.tapping:
            for phi in self.params.templates[0].e_means:
                
                
                
                # Add a tap event if it's that time
                if np.mod(np.angle(alpha_prev) - phi + np.pi, TWO_PI) < np.pi and np.mod(np.angle(alpha_prev) - phi + np.pi, TWO_PI) > np.pi/2 and np.mod(np.angle(alpha) - phi + np.pi, TWO_PI) > np.pi :
                    
                    if len(self.streams[0].e_times_p)==0 or self.ts[t_i] - self.streams[0].e_times_p[-1] > .2:
                    
                        n_taps = self.streams[0].e_times_p.size
                        if self.params.verbose:
                            print('n_taps', n_taps)
                    
                        self.streams[0].e_times_p = np.insert(self.streams[0].e_times_p, n_taps, self.ts[t_i])
        
        z = super().step(t_i, z_prev, mu_prev, V_prev)
        
        if self.tapping:
            z = z + self.params.dt * self.params.movement_precision * (alpha_prev - z_prev)
            ##### DO THE MATH

            
        return z, alpha

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            alpha_prev = self.alpha_s[i-1]
            z, alpha = self.step(i, z_prev, mu_prev, V_prev, alpha_prev)
            mu, V = PIPPETStream.z_mu_V(z)
            # Noise
            
            if self.ts[i] > self.params.start_tapping and self.tapping==False:
                self.tapping=True
                #alpha = z/np.abs(z)
                
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = z
            self.alpha_s[i]=alpha


class tappingPIPPET(cPIPPET):
    '''PIPPET with dynamic tapping movement '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.finger_s = np.ones(self.n_ts, dtype =np.double)
        
        self.tapping = False
        #self.params.delay_timesteps = int(self.params.delay / self.params.dt)


    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float, finger_prev: float):
        ''' Posterior update for a time step '''
        
        current_tap_stability = self.params.tap_stability * self.streams[0].zlambda(mu_prev, V_prev, self.params.omega)
        finger = finger_prev + self.params.dwell_stability * (1 - finger_prev) * self.params.dt + current_tap_stability * (self.params.tap_target - finger_prev) * self.params.dt
        
        if self.tapping and finger_prev>0 and finger < 0 and (len(self.streams[0].e_times_p)==0 or self.ts[t_i] - self.streams[0].e_times_p[-1] > .2):
                    
                        n_taps = self.streams[0].e_times_p.size
                        if self.params.verbose:
                            print('n_taps', n_taps)
                    
                        self.streams[0].e_times_p = np.insert(self.streams[0].e_times_p, n_taps, self.ts[t_i])
        
        z = super().step(t_i, z_prev, mu_prev, V_prev)

            
        return z, finger

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            finger_prev = self.finger_s[i-1]
            z, finger = self.step(i, z_prev, mu_prev, V_prev, finger_prev)
            mu, V = PIPPETStream.z_mu_V(z)
            # Noise
            
            if self.ts[i] > self.params.start_tapping and self.tapping==False:
                self.tapping=True
                #alpha = z/np.abs(z)
                
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = z
            self.finger_s[i]=finger



class gcPATIPPET(cPIPPET):
    ''' Oscillatory PIPPET Bank (gradient, circular)'''

    def __init__(self, params: PIPPETParams, omegas: np.ndarray, prior: np.ndarray=None):
        super().__init__(params)
        self.omegas = omegas
        self.omega_centers = (omegas[1:] + omegas[:-1]) / 2
        self.domega_list = np.diff(omegas)
        self.n_bank = self.omega_centers.shape[0]
        # TODO
        self.L_s = np.zeros(self.n_ts)
        self.L_ms = np.zeros((self.n_ts, self.n_bank))
        self.p_m = np.zeros((self.n_ts, self.n_bank))
        if prior is None:
            prior = np.ones(self.n_bank)
        self.p_m[0] = prior
        self.p_m[0] = self.p_m[0]/(self.p_m[0] * self.domega_list).sum()
        # Initialise big Lambdas using mu_0 and V_0
        for i, omega in enumerate(self.omega_centers):
            self.L_ms[0, i] = self.streams[0].zlambda(self.mu_s[0], self.V_s[0], omega)
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0] * self.domega_list)
        # TODO
        self.z_ms = np.ones((self.n_ts, self.n_bank), dtype=np.clongdouble)
        self.z_ms[0, :] = self.z_s[0]
        self.y_ms = np.ones((self.n_ts, self.n_bank), dtype=np.clongdouble)
        self.y_ms[0, :] = self.z_ms[0] * self.p_m[0] #* np.diff(self.omegas)
        # TODO
        self.mu_ms = np.zeros((self.n_ts, self.n_bank))
        self.mu_ms[0] = self.mu_s[0]
        self.V_ms = np.zeros((self.n_ts, self.n_bank))
        self.V_ms[0] = self.V_s[0]
        self.integrated = np.zeros(self.n_ts, dtype=np.clongdouble)
        self.mu_avg = np.zeros(self.n_ts)
        self.V_avg = np.zeros(self.n_ts)
        self.omega_avg = np.zeros(self.n_ts)
        self.integrated[0] = np.sum(self.y_ms[0, :]*self.domega_list.astype(complex))
        self.mu_avg[0] = np.angle(self.integrated[0])
        self.V_avg[0] = -2*np.log(np.abs(self.integrated[0]))
        self.omega_avg[0] = np.sum(self.p_m[0,:]*self.omega_centers*self.domega_list)

    def step(self, t_i: float) -> complex:
        ''' Posterior update for a time step '''
        try:
            t_j = t_i - 1
            t_prev, t = self.ts[t_j], self.ts[t_i]
            is_event = self.is_onset(t_prev, t, 0)

            ys_prev = self.y_ms[t_j]
            ps_prev = self.p_m[t_j]
            lams_prev = self.L_ms[t_j]
            lam_prev = self.L_s[t_j]

            for i, omega in enumerate(self.omega_centers):
                y_prev = ys_prev[i]
                p_prev = ps_prev[i]
                domega = self.omegas[i+1] - self.omegas[i]
                z_prev = y_prev/p_prev

                mu_prev, V_prev = PIPPETStream.z_mu_V(z_prev)

                p_flux_up = 0
                y_flux_up = 0
                if i != (self.n_bank-1):
                    p_flux_up = -(self.params.sigma_omega**2)/2
                    p_flux_up *= (ps_prev[i+1]-ps_prev[i])/(self.omega_centers[i+1]-self.omega_centers[i])
                    y_flux_up = -(self.params.sigma_omega**2)/2
                    y_flux_up *= (ys_prev[i+1]-ys_prev[i])/(self.omega_centers[i+1]-self.omega_centers[i])
                    #
                    p_flux_up += (ps_prev[i+1]+ps_prev[i])/2 * (self.params.omega_p - self.omegas[i+1]) * self.params.omega_p_tendency
                    y_flux_up += (ys_prev[i+1]+ys_prev[i])/2 * (self.params.omega_p - self.omegas[i+1]) * self.params.omega_p_tendency

                p_flux_down = 0
                y_flux_down = 0
                if i != 0:
                    p_flux_down =  -(self.params.sigma_omega**2)/2
                    p_flux_down *= (ps_prev[i]-ps_prev[i-1])/(self.omega_centers[i]-self.omega_centers[i-1])
                    y_flux_down =  -(self.params.sigma_omega**2)/2
                    y_flux_down *= (ys_prev[i]-ys_prev[i-1])/(self.omega_centers[i]-self.omega_centers[i-1])
                    #
                    p_flux_down += (ps_prev[i-1]+ps_prev[i])/2 * (self.params.omega_p - self.omegas[i]) * self.params.omega_p_tendency
                    y_flux_down += (ys_prev[i-1]+ys_prev[i])/2 * (self.params.omega_p - self.omegas[i]) * self.params.omega_p_tendency
                
                p_flux_out = self.params.sigma_omega_global * ps_prev[i]
                p_flux_in = self.params.sigma_omega_global * np.sum(ps_prev) / len(self.omega_centers)
                
                dp_c = (p_flux_down - p_flux_up + p_flux_in - p_flux_out)/domega
                dy_c = (y_flux_down - y_flux_up  - ys_prev[i]/ps_prev[i] * p_flux_out)/domega

                y_hat = p_prev * self.streams[0].z_hat(mu_prev, V_prev, lams_prev[i], omega) * lams_prev[i]/lam_prev
                p_hat = p_prev * lams_prev[i]/lam_prev

                y_hat = y_hat / np.maximum(1, abs(y_hat/p_hat)+.0001) ### numerical problem fixer



                #### make lambdas n_stream dimensional
                ## Introduce yhat contin effect one stream at a time
                ## if event goes outside oscillator loop, does another whole loop, revising CURRENT lambda values

                
                # Coupling, p
                dp = self.params.dt * dp_c * self.params.omega_c_coef
                dp -= self.params.dt * lam_prev*(p_hat - p_prev)
                p = p_prev + dp
                

                # Coupling, y
                dy = self.params.dt * dy_c * self.params.omega_c_coef
                # Drift
                dy += self.params.dt * y_prev*complex(-(self.params.sigma_phi**2)/2, omega)
                
                
                # Expectation
                if self.params.continuous_expectation:
                    y = y_prev + dy - lam_prev*(y_hat-y_prev)*self.params.dt
                else:
                    y = y_prev + dy
                
                
                if is_event:
                    y = y_hat
                    p = p_hat
                
                p = np.maximum(p, 0.00001)
                y = y / np.maximum(1, abs(y/p)+.0001) ### numerical problem fixer

                mu, V = PIPPETStream.z_mu_V(y/p)
                if V<0:
                    print("V = ", V)
                    print(np.abs(y))
                    print(p)
                    print(np.abs(y/p))
                    
                self.mu_ms[t_i, i] = mu
                self.V_ms[t_i, i] = V
                self.L_ms[t_i, i] = self.streams[0].zlambda(mu, V, omega)
                self.y_ms[t_i, i] = y
                self.p_m[t_i, i] = p

            self.integrated[t_i] = np.sum(self.y_ms[t_i,:]*self.domega_list.astype(complex))
            self.mu_avg[t_i] = np.angle(self.integrated[t_i])
            self.V_avg[t_i] = -2*np.log(np.abs(self.integrated[t_i]))
            self.omega_avg[t_i] = np.sum(self.p_m[t_i,:]*self.omega_centers*self.domega_list)

            self.L_s[t_i] = np.sum(self.domega_list * self.p_m[t_i] * self.L_ms[t_i])

            if is_event:
                self.event_n[0] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(0)
        except RuntimeWarning:
            print("y_prev =", y_prev, " p_prev =", p_prev, " zprev =", np.abs(y_prev/p_prev), " and ", self.streams[0].z_hat(mu_prev, V_prev, lams_prev[i], omega))
            breakpoint()

    def run(self) -> None:
        for i in range(1, self.n_ts):
            self.step(i)
            # TODO: noise!

            

class vcPATIPPET(cPIPPET):
    '''Variational circular phase and tempo inference'''
    
    
    def __init__(self, params: PIPPETParams):
        
        new_templates = []
        for p in params.templates:
            new_means = np.append(p.e_means,0)
            new_vars = np.append(p.e_vars,100)
            new_lambdas = np.append(p.e_lambdas, p.lambda_0)
            new_templates.append(TemplateParams(p.e_times, new_means, new_vars, new_lambdas, p.lambda_0, p.label))
        params.templates = new_templates
        
        super().__init__(params)
        
        self.V_omegas = np.zeros(self.n_ts)
        self.V_omegas[0] = self.params.V_omega_0

        self.V_zomegas = np.zeros(self.n_ts, dtype=np.clongdouble)
        self.V_zomegas[0] = self.params.S_0 * np.exp(1j*self.params.mu_0 - self.params.V_0/2)* self.params.V_omega_0
        

        self.S = np.ones(self.n_ts)
        self.S[0] = self.params.S_0 

        self.Omegas = np.zeros((self.n_ts,))
        self.Omegas[0] = self.params.omega_0

        self.L_s = np.zeros((self.n_ts,))
        self.L_s[0] = np.nan
    
        self.M = np.arange(-self.params.m_span, self.params.m_span+1, 1)
        self.N = np.arange(1, self.params.m_span+1, 1)
        self.oM = np.ones(np.size(self.M))
        self.oN = np.ones(np.size(self.N))

    def step(self, t_i: float) -> complex:

        # Previous time step values
        z_prev = self.z_s[t_i-1]
        mu_prev = self.mu_s[t_i-1]
        V_prev = self.V_s[t_i-1]
        Omega_prev = self.Omegas[t_i-1]
        S_prev = self.S[t_i-1]
        V_omega_prev = self.V_omegas[t_i-1]
        V_zomega_prev = self.V_zomegas[t_i-1]

        # Event?
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        is_event = self.is_onset(t_prev, t, 0)
        dNt = int(is_event)

        #blambda = self.streams[0].zlambda_tempo(mu_prev, V_prev, V_omega_prev, S_prev)
        #z_hat = self.streams[0].z_hat_tempo(mu_prev, V_prev, V_omega_prev, S_prev, blambda)
        #omega_hat = self.streams[0].omega_hat(mu_prev, V_prev, V_omega_prev, S_prev, Omega_prev, blambda)

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_par += -S_prev * V_omega_prev * self.params.dt
        dz_perp = Omega_prev * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par)
        if dz_par>0 and self.params.verbose:
            print('dz_par is positive ({} at {})'.format(dz_par, self.ts[t_i]))
        

        dOmega =  self.params.omega_p_tendency*(self.params.omega_p - Omega_prev)*self.params.dt
        Omega = Omega_prev + dOmega
        dV_omega = self.params.sigma_omega**2 / 2 * self.params.dt
        V_omega = V_omega_prev * np.exp(-2 * self.params.omega_p_tendency * self.params.dt) + dV_omega

        dV_zomega_perp = Omega_prev * self.params.dt
        V_zomega = V_zomega_prev * np.exp(1j*dV_zomega_perp  - self.params.omega_p_tendency * self.params.dt)
        V_zomega += (V_omega_prev - S_prev**2 * V_omega_prev**2 - self.params.omega_p_tendency*(self.params.omega_p-Omega_prev))* z_prev*1j * self.params.dt
        
        S = (V_zomega/(1j* z * V_omega)).real
        
        if abs(z)<0.0001:
            S = S_prev
            V_zomega = V_zomega_prev
            z = z_prev


        mu = np.angle(z)
        V = -2*np.log(np.abs(z))
            
        if self.params.tempo_scaling:
            v = self.streams[0].params.e_vars * Omega_prev**2
                
        else:
            v = self.streams[0].params.e_vars
            
        
        C_0_0 = -(self.M**2) * ((V+v)/2).reshape(-1, 1) \
               - 1j * self.M * (mu-self.streams[0].params.e_means).reshape(-1, 1) #\
                   
        C_0_ = (self.streams[0].params.e_lambdas/TWO_PI).reshape(-1, 1) * np.exp(C_0_0)
        
        
        blambda_i = np.sum(C_0_, 1).real
        blambda = np.sum(blambda_i)
        
        if is_event:
            if self.params.verbose:
                print('event')
                print(t)
            self.event_n[0] += 1
            self.idx_event.add(t_i)
            self.event_stream[t_i].add(0)
            

            C_1_ = -(self.M**2) * (V/2) \
                   -(self.M+1)**2 * (v/2).reshape(-1, 1) \
                   - 1j * self.M * mu \
                   + 1j * (self.M+1) * self.streams[0].params.e_means.reshape(-1, 1)

            C_1_ = (self.streams[0].params.e_lambdas/TWO_PI).reshape(-1, 1) * np.exp(C_1_)
            
            
            z_hat = (1/blambda) * C_1_.sum()
            
            if abs(z_hat)>1:
                print('z_hat is too big ({} at {})'.format(abs(z_hat), self.ts[t_i]))
                z_hat = z_hat/abs(z_hat)
                
            
            omega_hat = Omega - (S * 1j*V_omega/blambda * (self.M * C_0_).sum()).real
            


            Omega_plus = omega_hat
            S_hat = max(0, (S  * -((self.M.reshape(1, -1)) * C_1_).sum()/blambda).real)
            
            
            
            V_omega_hat = V_omega - (Omega - Omega_plus)**2
            
            V_omega_hat += -(S**2*V_omega**2/blambda) * (self.M**2 * C_0_).sum()
            
            V_omega_hat = max(V_omega_hat, 0)
            
            Omega = Omega_plus
            z = z_hat
            #assert abs(z_hat)<1, 'z_hat is too big ({} at {})'.format(abs(z_hat), self.ts[t_i])
            
            V_omega = np.maximum(0, V_omega_hat.real)
            
            V_zomega_hat =  S_hat * 1j * V_omega_hat * z_hat
            V_zomega = V_zomega_hat
            S = S_hat
            
        self.z_s[t_i] = z
        #assert abs(z)<1, 'z is too big ({} at {})'.format(abs(z_hat), self.ts[t_i])
        self.Omegas[t_i] = Omega
        self.S[t_i] = S
        self.V_zomegas[t_i] = V_zomega
        self.V_omegas[t_i] = V_omega
        self.L_s[t_i] = blambda

    def run(self) -> None:
        for i in range(1, self.n_ts):
            self.step(i)
            mu, V = PIPPETStream.z_mu_V(self.z_s[i])
            # TODO: add noise
            pass
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = np.exp(complex(-V/2, mu))
            

if __name__ == "__main__":
    import pdb
    print('Debugger on - press \'c\' to continue examples, \'q\' to quit')
