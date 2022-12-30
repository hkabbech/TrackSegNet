import numpy as np
import pandas as pd

def compute_ptm(df, par):
    counts = np.zeros((par['num_states'], par['num_states']), dtype=int)
    for N in tqdm(df['track_id'].unique()):
        track = df[df['track_id'] == N]
        for i, j in zip(track['state'], track['state'][1:]):
            if pd.notna(i) and pd.notna(j):
                # print(i, j)
                counts[int(i)][int(j)] += 1
    # estimate Probs
    sum_of_rows = counts.sum(axis=1)
    # if some row-sums are zero, replace with any number becasue the sum in in denomimator and
    # the numerator will be 0 in any case, so we will get the correct 0 for p_ij in the result
    sum_of_rows[sum_of_rows == 0] = 1
    pis = counts / sum_of_rows[:, None]
    ptm = pd.DataFrame(pis)
    ptm['num'] = sum_of_rows
    return ptm