'''
'''
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy.stats import norm

TWO_PI = 2*np.pi

def template(model: PIPPET) -> np.ndarray:
    ''' Collapse expectation templates into a numpy array '''
    from scipy.stats import norm
    from PIPPET import cPIPPET

    if isinstance(model, cPIPPET):
        # Wrapped expectations
        ts = np.arange(-np.pi, np.pi, model.params.dt)
        temp = np.zeros((model.n_streams, ts.size))
    else:
        # Expectations along a line
        ts = model.ts
        temp = np.zeros((model.n_streams, model.n_ts))

    for s_i in range(model.n_streams):
        stream = model.streams[s_i].params
        temp[s_i] += stream.lambda_0
        for i in range(stream.e_means.size):
            pdf = norm.pdf(ts, loc=stream.e_means[i], scale=(stream.e_vars[i])**0.5)
            temp[s_i] += stream.e_lambdas[i] * pdf
        if isinstance(model, cPIPPET):
            pdf = norm.pdf(ts, loc=stream.e_means[i]-np.pi, scale=(stream.e_vars[i])**0.5)
            temp[s_i] += stream.e_lambdas[i] * pdf

    return ts, temp


def merge_legends(axs):
    # Consolidated legend
    def _by_label(a):
        handles, labels = a.get_legend_handles_labels()
        return dict(zip(labels, handles))
    by_label = dict()
    for ax in axs:
        by_label.update(_by_label(ax))
    return by_label

def plot_template(model, ax_temp, cols, xlabelkwargs={}, ylabelkwargs={}):
    # Expectations
    ts, temps = template(model)
    for i, temp in enumerate(temps):
        ax_temp.plot(temp, ts, c=cols[i], label=model.labels[i], alpha=0.75)
    ax_temp.set_xlabel('Expectation λ(Φ)', **xlabelkwargs)
    ax_temp.set_ylabel('Phase Φ', **ylabelkwargs)

    
def plot_phase(model, ax, col, xlabelkwargs={}):
    # Phase progress
    std = 2*np.sqrt(model.V_s)
    ax.plot(model.ts, model.mu_s, c=col, linewidth=1.75, label='Est. phase, '+r'$\mu_t$')
    ax.fill_between(model.ts, model.mu_s-std, model.mu_s+std, alpha=0.2, facecolor=col, label='Est. var., '+r'$4\sqrt{V_t}$')
    ax.set_xlabel('Time', **xlabelkwargs)
    
def plot_events(model, ax, ax_temp, ax_prob=None, cols=None, stimulus=True, expected=True):
    # Stimulus/Auditory events
    if stimulus:
        for i in set(model.idx_event):
            col = {'black'} if not cols or not model.event_stream else {cols[s_i] for s_i in model.event_stream[i]}
            for c in col:
                ax.axvline(model.ts[i], color=c, alpha=0.75, linestyle='--', linewidth=1)
                if ax_prob:
                    ax_prob.axvline(model.ts[i], color=c, alpha=0.55, linestyle='-', linewidth=1)
    # Expected events
    if expected:
        for stream in model.streams:
            for e_m in stream.params.e_means:
                ax.axhline(e_m, color='grey', alpha=0.55, linestyle='--', linewidth=1)
                ax_temp.axhline(e_m, color='grey', alpha=0.55, linestyle='--', linewidth=1)

def plot_mPIPPET(model, wippet=False, xmax=None, title='', figsize=(8, 3)):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib import pyplot as plt
    from seaborn import color_palette
    
    cs = color_palette()

    # Create the grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2,  width_ratios=(1, 4))
    ax = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[0, 0], sharey=ax)

    plot_phase(model, ax, cs[0])
    plot_template(model, ax_temp, cs[1:])
    plot_events(model, ax, ax_temp, cols=cs[1:])
    
    ax.set_xlim([model.params.t0, model.tmax if not xmax else xmax])
    
    if wippet:
        ax.set_ylim([-np.pi, np.pi])
        #ax.axhline(0.0, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(0.0, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        ax.axhline(-np.pi/2, color='grey', alpha=0.33, linestyle='--', linewidth=1)
        ax.axhline(np.pi/2, color='grey', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(-np.pi/2, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(np.pi/2, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #for i in range(1, int(model.tmax/TWO_PI)+1):
        #    ax.axvline(i*TWO_PI, color='blue', alpha=0.55, linestyle='--', linewidth=1) 
    else:
        ax.set_ylim([model.params.t0, model.tmax])

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_temp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax_temp.tick_params(axis='both', which='major', labelsize=8)
    
    legend = merge_legends([ax, ax_temp])
    fig.legend(legend.values(), legend.keys(), loc='upper center', bbox_to_anchor=(0.65, 1.10),
               prop={'size':6}, ncol=2, fancybox=False, framealpha=1, edgecolor='black')
    
    if title:
        ax_temp.set_title(title)
    
    fig.tight_layout()
    return fig, ax_temp, ax


def plot_movingPIPPET(model, wippet=False, xmax=None, title='', figsize=(8, 3)):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib import pyplot as plt
    from seaborn import color_palette
    
    cs = color_palette()

    # Create the grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2,  width_ratios=(1, 4))
    ax = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[0, 0], sharey=ax)

    plot_phase(model, ax, cs[0])
    
    ax.plot(model.ts, np.angle(model.alpha_s), c=cs[1], linewidth=1.75, label='Movement cycle')
    plot_template(model, ax_temp, cs[1:])
    plot_events(model, ax, ax_temp, cols=cs[1:])
    
    ax.set_xlim([model.params.t0, model.tmax if not xmax else xmax])
    
    if wippet:
        ax.set_ylim([-np.pi, np.pi])
        #ax.axhline(0.0, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(0.0, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        ax.axhline(-np.pi/2, color='grey', alpha=0.33, linestyle='--', linewidth=1)
        ax.axhline(np.pi/2, color='grey', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(-np.pi/2, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #ax_temp.axhline(np.pi/2, color='blue', alpha=0.33, linestyle='--', linewidth=1)
        #for i in range(1, int(model.tmax/TWO_PI)+1):
        #    ax.axvline(i*TWO_PI, color='blue', alpha=0.55, linestyle='--', linewidth=1) 
    else:
        ax.set_ylim([model.params.t0, model.tmax])

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_temp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax_temp.tick_params(axis='both', which='major', labelsize=8)
    
    legend = merge_legends([ax, ax_temp])
    fig.legend(legend.values(), legend.keys(), loc='upper center', bbox_to_anchor=(0.65, 1.10),
               prop={'size':6}, ncol=2, fancybox=False, framealpha=1, edgecolor='black')
    
    if title:
        ax_temp.set_title(title)
    
    fig.tight_layout()
    return fig, ax_temp, ax

def plot_format_dist(ax, omega_centers, omega_lines, ymax, scale=1.1):
    ax.set_yscale('linear')
    #ax.set_xscale('log')
    ax.set_ylabel(r'Probability, $P^i_t$')
    ax.set_xlabel(r'Tempo, $\omega^i$')
    ax.set_xlim([omega_centers[0], omega_centers[-1]])
    ax.vlines(omega_lines, 0, ymax*scale, ls='--', color='k', alpha=0.1)
    ax.set_xticks(omega_lines)
    ax.set_xticklabels(omega_lines)

    
def get_masked_signal(signal, p_mask, p_thresh):
    hipass_signal = copy.deepcopy(signal)
    lopass_signal = copy.deepcopy(signal)
    for i in range(np.size(signal, 1)):
        hipass_signal[p_mask[:,i]<p_thresh, i]=np.nan
        lopass_signal[p_mask[:,i]>p_thresh, i]=np.nan
        hipass_signal[np.asarray(np.abs(hipass_signal[1:, i]-hipass_signal[:-1, i])>np.pi).nonzero()] =np.nan
        lopass_signal[np.asarray(np.abs(lopass_signal[1:, i]-lopass_signal[:-1, i])>np.pi).nonzero()] =np.nan
    
    return (hipass_signal, lopass_signal)
    
def plot_mus(ax, m, cs, p_thresh=.4, from_i=0, to_i=None, alpha=1):
    (hipass_mus, lopass_mus) = get_masked_signal(m.mu_ms, m.p_m, p_thresh)
    
    if to_i is None:
        to_i = m.ts.shape[0]
    for i, omega in enumerate(m.omega_centers):
        ax.plot(m.ts[from_i:to_i], hipass_mus[from_i:to_i, i], c=cs[i], alpha=alpha)
        ax.plot(m.ts[from_i:to_i], lopass_mus[from_i:to_i, i], c=cs[i], alpha=0.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Estim.'+'\n'+r'phases $\mu_t^i$')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_ylim([-np.pi, np.pi])
    
def plot_avg(ax, m, cs, from_i=0, to_i=None, alpha=1):
    if to_i is None:
        to_i = m.ts.shape[0]
        
    std = 2*np.sqrt(np.maximum(m.V_avg[from_i:to_i], 0))
    plot_w_jumps(ax, m.mu_avg, m.ts, from_i, to_i, cs)
    ax.fill_between(m.ts[from_i:to_i], m.mu_avg[from_i:to_i]-std, m.mu_avg[from_i:to_i]+std, alpha=0.2, facecolor=cs[0],
                    label='Estim. variance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Expected phase \n and uncertainty')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_ylim([-np.pi, np.pi])
    
def plot_omega_avg(ax, m, cs, from_i=0, to_i=None, tempo_range = None, alpha=1):
    if to_i is None:
        to_i = m.ts.shape[0]
    tempi = m.omega_avg / TWO_PI *60
    print(tempi)
    ax.plot(m.ts[from_i:to_i], tempi[from_i:to_i], c=cs[0], label='Est. tempo (bpm)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Estim. \n tempo (bpm)')
    if tempo_range != None:
        ax.set_ylim(tempo_range)
    
def plot_Vs(ax, m, cs, p_thresh=.4, from_i=0, to_i=None, alpha=0.6):
    (hipass_Vs, lopass_Vs) = get_masked_signal(m.V_ms, m.p_m, p_thresh)
    
    if to_i is None:
        to_i = m.ts.shape[0]
    for i, omega in enumerate(m.omega_centers):
        ax.plot(m.ts[from_i:to_i], hipass_Vs[from_i:to_i, i], c=cs[i], alpha=alpha)
        ax.plot(m.ts[from_i:to_i], lopass_Vs[from_i:to_i, i], c=cs[i], alpha=0.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Phase uncertainty $V_t^i$')
    ax.set_ylim([0,.2])
    
def plot_ps(ax, m, cs, from_i=0, to_i=None, alpha=0.6):
    if to_i is None:
        to_i = m.ts.shape[0]
    for i, omega in enumerate(m.omega_centers):
        ax.plot(m.ts[from_i:to_i], m.p_m[from_i:to_i, i], c=cs[i], alpha=alpha)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Conditional '+'\n'+r'probabilities $P_t^i$')

def plot_Ls(ax, m, cs, from_i=0, to_i=None, alpha=1):
    if to_i is None:
        to_i = m.ts.shape[0]
    #log_L = np.log(np.maximum(m.streams[0].params.lambda_0, m.L_s[from_i:to_i]))
    ax.plot(m.ts[from_i:to_i], m.L_s[from_i:to_i], alpha=alpha) #np.log
    ax.set_xlabel('Time (s)')
    #ax.set_ylabel(r'Log hazard'+'\n'+r'rate $\log(\Lambda_t)$')
    ax.set_ylabel(r'Hazard rate $\Lambda_t$')

def plot_gcPATIPPET(m, start_event, focal_event, end_event, focal_label, p_thresh, tempo_range, preroll = 100, postroll = 200):
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(4, 1)
    ax_mu = fig.add_subplot(gs[0, 0])
    #ax_V = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_tempo = fig.add_subplot(gs[2, 0])
    ax_L = fig.add_subplot(gs[3, 0])
    e_is = list(sorted(m.idx_event))

    
    if start_event == -1:
        from_i = 0
        start_event = 0
    else:
        from_i=e_is[start_event]-preroll
        
    
    if end_event == -1:
        to_i=-1
        end_event = len(e_is)-1
    else:
        to_i=e_is[end_event]+postroll
    omega_centers = m.omega_centers
    omega_lines = [.8, .9, 1, 1.2]
    omega_n = len(omega_centers)

    cs = sns.color_palette('bright', 2)
    #cs_mus = plt.get_cmap("Greys", omega_n)
    cs_mus = sns.color_palette('viridis', omega_n)[::-1]

    final_t = m.ts[e_is[-1]]
    
    
    plot_mus(ax_mu, m, cs_mus, p_thresh = p_thresh, from_i=from_i, to_i=to_i)

    
    #plot_Vs(ax_V, m, cs_mus, p_thresh = p_thresh, from_i=from_i, to_i=to_i)

    plot_ps(ax_p, m, cs_mus, from_i=from_i, to_i=to_i)

    #plot_avg(ax_avg, m, cs, from_i=from_i, to_i=to_i)
    
    plot_omega_avg(ax_tempo, m, cs, from_i=from_i, to_i=to_i, tempo_range = tempo_range)
    plot_Ls(ax_L, m, cs, from_i=from_i, to_i=to_i)
    
    for ax in [ax_mu, ax_p, ax_tempo, ax_L]:
        for i in range(start_event, end_event+1):
            if i==focal_event:
                ax.axvline(m.ts[e_is[i]], -np.pi, np.pi, c='red', lw=2, ls='--', label=focal_label)
            else:
                ax.axvline(m.ts[e_is[i]], -np.pi, np.pi, c='orange', lw=2, ls='--')
    return fig

def plot_w_jumps(ax, mu, t, from_i, to_i, cs):
    mu_tmp = mu
    mu_tmp[np.asarray(np.abs(mu_tmp[1:]-mu_tmp[:-1])>np.pi).nonzero()] =np.nan
    
    ax.plot(t[from_i:to_i], mu[from_i:to_i], c=cs[0], linewidth=1.75, label='Est. phase')
    


def plot_vcPATIPPET(m,start_event, focal_event, end_event, focal_label, tempo_range, preroll = 100, postroll = 200):
    e_is = list(sorted(m.idx_event))
    
    if start_event == -1:
        from_i = 0
        start_event = 0
    else:
        from_i=e_is[start_event]-preroll
        
    
    if end_event == -1:
        to_i=-1
        end_event = len(e_is)-1
    else:
        to_i=e_is[end_event]+postroll
    cs = sns.color_palette('bright', 2)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(4, 1)
    ax_mu = fig.add_subplot(gs[0, 0])
    ax_S = fig.add_subplot(gs[1, 0])
    ax_tempo = fig.add_subplot(gs[2, 0])
    ax_L = fig.add_subplot(gs[3, 0])
    
    std = 2*np.sqrt(np.maximum(m.V_s[from_i:to_i], 0))
    omega_std = 2*np.sqrt(m.V_omegas[from_i:to_i]) / TWO_PI *60
    tempi = m.Omegas / TWO_PI *60
    
    plot_w_jumps(ax_mu, m.mu_s, m.ts, from_i, to_i, cs)
    
    ax_mu.fill_between(m.ts[from_i:to_i], m.mu_s[from_i:to_i]-std, m.mu_s[from_i:to_i]+std, alpha=0.2, facecolor=cs[0],
                    label='Estim. variance')
    ax_mu.set_yticks([-np.pi, 0, np.pi])
    ax_mu.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax_mu.set_ylim([-np.pi, np.pi])
    ax_mu.set_ylabel(r'Estim. phase $\mu_t$') 
      #axs[1].plot(m.ts, abs(m.z_s))
      #axs[1].set_title(r'$|z|$')
    
    
    
    ax_tempo.plot(m.ts[from_i:to_i], tempi[from_i:to_i], linewidth=1.75, label='Est. tempo')
    ax_tempo.fill_between(m.ts[from_i:to_i], tempi[from_i:to_i]-omega_std, tempi[from_i:to_i]+omega_std, alpha=0.2, facecolor=cs[0],
                    label='Estim. tempo variance')
    #axs[1].set_title(r'$\bar{\omega}$')
    ax_tempo.set_ylabel(r'Estim. tempo'+'\n'+r'$\omega_t$ (bpm)')
    if tempo_range != None:
        ax_tempo.set_ylim(tempo_range)

    ax_S.plot(m.ts[from_i:to_i], m.S[from_i:to_i].real)
    ax_S.set_ylabel(r'Phase/tempo '+'\n'+r'dependence $S$')
    

    
    plot_Ls(ax_L, m, cs, from_i=from_i, to_i=to_i)
    ax_L.set_xlabel('Time (s)')
    for ax in [ax_mu, ax_tempo, ax_S, ax_L]:
        for i in range(start_event, end_event+1):
            if i==focal_event:
                ax.axvline(m.ts[e_is[i]], -np.pi, np.pi, c='red', lw=2, ls='--', label=focal_label)
            else:
                ax.axvline(m.ts[e_is[i]], -np.pi, np.pi, c='orange', lw=2, ls='--')

    return fig
    
def _plot_template(m, omega, figsize=(7, 2.5)):
    # Templates
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    gs = fig.add_gridspec(1, 2)
    ax_lam = fig.add_subplot(gs[0, 0])
    ax_loglam = fig.add_subplot(gs[0, 1])
    ts = np.arange(-np.pi, np.pi, m.params.dt)
    stream = m.streams[0].params
    temp = stream.lambda_0 + np.zeros(np.shape(stream.e_lambdas[0]))
    for i in range(stream.e_means.size):
        pdf = norm.pdf(ts, loc=stream.e_means[i], scale=(stream.e_vars[i])**0.5 * omega)
        pdf2 = norm.pdf(ts, loc=stream.e_means[i]-TWO_PI, scale=(stream.e_vars[i])**0.5 * omega)
        
        temp = temp + stream.e_lambdas[i]*pdf + stream.e_lambdas[i]*pdf2
        
    #ax_temp.plot(np.log(temp), ts, c=cs[1], label=m.labels[0])
    #ax_temp.set_xlabel('Log Expectation log(λ(Φ))')
    ax_lam.plot(ts, (temp), c="black", label=m.labels[0])
    ax_lam.set_ylabel('Expectation λ(Φ)')
    ax_lam.set_xlabel('Phase Φ')
    ax_lam.set_xticks([-np.pi, 0, np.pi])
    ax_lam.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    ax_loglam.plot(ts, np.log(temp), c="black", label=m.labels[0])
    ax_loglam.set_ylabel('Log Expectation\n log(λ(Φ))')
    ax_loglam.set_xlabel('Phase Φ')
    ax_loglam.set_xticks([-np.pi, 0, np.pi])
    ax_loglam.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    plt.tight_layout()
    return fig

def _plot_log_template(m, figsize=(4, 2)):
    # Templates
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ts = np.arange(-np.pi, np.pi, m.params.dt)
    stream = m.streams[0].params
    temp = stream.lambda_0 + np.zeros(np.shape(stream.e_lambdas[0]))
    for i in range(stream.e_means.size):
        pdf = norm.pdf(ts, loc=stream.e_means[i], scale=(stream.e_vars[i])**0.5 * TWO_PI/.63)
        pdf2 = norm.pdf(ts, loc=stream.e_means[i]-TWO_PI, scale=(stream.e_vars[i])**0.5* TWO_PI/.63)
        
        temp = temp + stream.e_lambdas[i]*pdf + stream.e_lambdas[i]*pdf2
        
    #ax_temp.plot(np.log(temp), ts, c=cs[1], label=m.labels[0])
    #ax_temp.set_xlabel('Log Expectation log(λ(Φ))')
    ax.plot(ts, np.log(temp), c="black", label=m.labels[0])
    ax.set_ylabel('Log Expectation log(λ(Φ))')
    ax.set_xlabel('Phase Φ')
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    return fig


def plot_cPIPPET(m, figsize=(12, 5), width_ratios=(1, 5), title='', xlim=None):
    e_is = list(sorted(m.idx_event))
    
    from_i=1
    to_i=len(m.ts)
    
    cs = sns.color_palette('bright', 2)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2,  width_ratios=width_ratios)
    ax = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[0, 0], sharey=ax)
    
    std = 2*np.sqrt(np.maximum(m.V_s[from_i:to_i], 0))
    
    plot_w_jumps(ax, m.mu_s, m.ts, from_i, to_i, cs)
    
    ax.fill_between(m.ts[from_i:to_i], m.mu_s[from_i:to_i]-std, m.mu_s[from_i:to_i]+std, alpha=0.2, facecolor=cs[0],
                    label='Est. variance')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_ylim([-np.pi, np.pi])

    

    for i in range(0, len(e_is)):
        ax.axvline(m.ts[e_is[i]], -np.pi, np.pi, c='orange', lw=2, ls='--')

    ts = np.arange(-np.pi, np.pi, m.params.dt)
    stream = m.streams[0].params
    temp = m.streams[0].params.lambda_0 + np.zeros(np.shape(stream.e_lambdas[0]))
    if m.params.tempo_scaling:
        v = stream.e_vars * m.params.omega **2
    else:
        v = stream.e_vars
        
    for i in range(stream.e_means.size):
        pdf = norm.pdf(ts, loc=stream.e_means[i], scale=(v[i])**0.5)
        pdf2 = norm.pdf(ts, loc=stream.e_means[i]-TWO_PI, scale=(v[i])**0.5)
        
        temp = temp + stream.e_lambdas[i]*pdf + stream.e_lambdas[i]*pdf2
        
    #ax_temp.plot(np.log(temp), ts, c=cs[1], label=m.labels[0])
    #ax_temp.set_xlabel('Log Expectation log(λ(Φ))')
    ax_temp.plot(np.log(temp), ts, c="black", label=m.labels[0])
    ax_temp.set_xlabel('Log Expectation log(λ(Φ))')
    ax_temp.set_ylabel('Phase Φ')
    ax.axhline(0.0, color='black', alpha=0.55, linestyle='--', linewidth=1)
    ax_temp.axhline(0.0, color='black', alpha=0.55, linestyle='--', linewidth=1)
    ax.axhline(-np.pi/2, color='black', alpha=0.25, linestyle='--', linewidth=1)
    ax.axhline(np.pi/2, color='black', alpha=0.25, linestyle='--', linewidth=1)
    ax_temp.axhline(-np.pi/2, color='black', alpha=0.55, linestyle='--', linewidth=1)
    ax_temp.axhline(np.pi/2, color='black', alpha=0.55, linestyle='--', linewidth=1)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([m.params.t0, m.tmax + 0.1])
    return fig