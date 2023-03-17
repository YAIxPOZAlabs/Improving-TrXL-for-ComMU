import numpy as np
import math
from scipy import spatial
from itertools import combinations

def extract_attr_vector_per_bar(note, attr: str = None):
    # set parameters by attribute
    dim, index_start, index_end = None, None, None
    if attr == 'chroma': # <NOTE PITCH>
        dim, index_start, index_end = 12, 3, 130
    elif attr == 'grv': # <DURATION>
        dim, index_start, index_end = 128, 304, 431

    total_out = []
    for n in note:
        if n == 2:
            out = [0] * dim
            total_out.append(out)
        if index_start <= n <= index_end:
            n_shift = n - index_start # shift to start from 0
            ind = n_shift % 12 if attr == "chroma" else n_shift
            out[int(ind)] += 1

    return total_out


def get_dist(note_1, note_2, attr: str = None):
    # compute attr vectors
    vec_1 = extract_attr_vector_per_bar(note_1, attr=attr)
    vec_2 = extract_attr_vector_per_bar(note_2, attr=attr)
    assert len(vec_1) == len(vec_2), "Inputs must have the same shape!"
    # compute cosine similarity between attr. vectors
    out = []
    for v1, v2 in zip(vec_1, vec_2):
        if sum(v1) * sum(v2) == 0:
            continue
        dist = spatial.distance.cosine(v1, v2)
        out.append(dist)

    return np.array(out).mean() # mean over bar


def get_diversity(notes):
    """
    INPUT: notes
    - type: numpy.ndarray
    - size: [M, n, 1012]
    - info: M개 메타정보에 대해, 각 메타정보 당 n개씩 생성된 노트 시퀀스 set

    OUTPUT: nanmean of dist
    - type: float
    - size: [,]
    - info: 전체 평균 diversity 값
    """
    dist = []
    for i in range(len(notes)):
        note_combination_lst = list(combinations(notes[i], 2))
        local_dist = []
        for combi_tuple in note_combination_lst:
            try:
                sim_chr = get_dist(combi_tuple[0], combi_tuple[1], attr='chroma')
                sim_grv = get_dist(combi_tuple[0], combi_tuple[1], attr='grv')
                numerator = np.square(sim_chr) + np.square(sim_grv)
                local_dist.append(math.sqrt(numerator) / 2)
            except:
                local_dist.append(float('nan'))
        # mean of n note sequences from the same metadata
        dist.append(np.nanmean(np.array(local_dist)))
    return np.nanmean(np.array(dist)) # mean over metadata


# if __name__ == "__main__":
# LOAD DATA
# real_npy = np.load("/media/data/dioai/preprocessed_data/paper_data/no_aug/output_npy/target_val.npy", allow_pickle=True)
# gen_npy = np.load("/media/data/dioai/preprocessed_data/paper_data/no_aug/exp_decoded_obj/200_95/00/generated.npy", allow_pickle=True)
n_gen = np.load("/workspace/ComMU-Transformer-conditional-VAE/out_val/val.npy", allow_pickle=True)
np_gen = np.zeros((763, 10, 1012))
i = 0
for ii in n_gen:
    j = 0
    for jj in ii:
        k = 0
        for kk in jj:
            if k >= 1012:
                break
            np_gen[i, j, k] = kk
            k += 1
        j += 1
    i += 1
n_gen = np_gen
################### dummy data for debugging ###################
# n_gen_raw = np.zeros([1, 10, 1024])
# np.random.seed(1456)
# for n in range(10):
#     n_gen_raw[0][n] = np.random.randint(3, 559, size=1024)
#     bar_len = 1012 // 4
#     for nn in range(4):
#         n_gen_raw[0][n][12 + bar_len * nn] = 2
    # print(len(np.where(n_gen_raw[0][n]==2)[0])) # 4 measures
################################################################

'''
n_gen.shape: [# of metadata, # of samples per metadata, sequence length]
'''
# for j in range(n_gen.shape[0]):
#     for k in range(10):
#         n_gen[j][k] = np.array(n_gen[j][k][12:])
n_gen = n_gen[:, :, 12:] # if numpy matrix

# COMPUTE METRICS
d = get_diversity(n_gen)
print(f"diversity: {d}")



