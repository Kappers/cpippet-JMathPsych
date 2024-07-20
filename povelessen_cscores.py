''' Simple C-score implementation.

Povel, D.-J., & Essens, P. (1985). Perception of Temporal Patterns.
Music Perception: An Interdisciplinary Journal, 2(4), 411–440. https://doi.org/10.2307/40285311

@Tom Kaplan: t.m.kaplan@qmul.ac.uk
'''
from __future__ import annotations
import numpy as np

def accent_pattern(iois: list[int]) -> np.ndarray:
    # [1, 2, 2, 1, ...] -> [1, 1, 0, 1, 0, 1, ...]
    iois_ts = np.array(sum(([1] + [0] * (x-1) for x in iois), []))
    iois_ts_acc = iois_ts.copy()

    # Rules (1-3) from p415
    def _accent(cluster):
        n = len(cluster)
        if n == 1:
            return [1]
        elif n == 2:
            return [0, 1]
        else:
            return [1] + [0]*(n-2) + [1]
        
    zeros = np.where(iois_ts == 0)[0]
    
    # First event is not a rest
    if zeros[0] > 0:
        cluster = iois_ts[:zeros[0]]
        iois_ts_acc[:zeros[0]] += _accent(iois_ts[:zeros[0]])

    for i, j in zip(zeros, zeros[1:]):
        if i+1 != j:
            iois_ts_acc[i+1:j] += _accent(iois_ts[i+1:j])

    # Last event isn't a rest
    if zeros[-1] < len(iois_ts)-1:
        cluster = iois_ts[zeros[-1]+1:]
        iois_ts_acc[zeros[-1]+1:] += _accent(iois_ts[zeros[-1]+1:])

    return iois_ts_acc

def cscores(iois: list[int], W: int) -> tuple[int, int, bool, int]:
    # "Time" representation of IOI sequence
    iois_ts = accent_pattern(iois)
    # Test all period/offset clock strengths
    for period in range(1, sum(iois)//2):
        for loc in range(1, period+1):
            iois_ts_shift = np.roll(iois_ts, -(loc-1))
            mask = iois_ts_shift[0::period]
            zero_ev = (mask == 1).astype(int).sum()
            m1_ev = (mask == 0).astype(int).sum()
            C = W*m1_ev + zero_ev
            divisible_period = not (sum(iois) % period)
            yield period, loc, divisible_period, C

def best_clock(iois: list[int], W: int) -> tuple[tuple[int, int], int]:
    best, best_C = (None, None), np.inf
    for period, loc, div, C in cscores(iois, W):
        if div and C < best_C:
            best_C = C
            best = (period, loc)
    return best, best_C

if __name__ == "__main__": 
    iois = [1, 2, 2, 1, 1, 2, 3]

    (period, loc), C = best_clock(iois, 4)
    print('-> {} with {}'.format((period, loc), C))

