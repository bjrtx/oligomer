import csv
import numpy as np

filenames = ...
thresholds = ...
hotspot_list = ...

chains_list = (f"Bfr_molmap_chain{char}_res5.mrc" for char in string.ascii_lowercase[:24])

def read_mrc(filename):
    with mrcfile.open(filename) as f:
        return f.data


for filename, threshold in zip(filenames, thresholds):
    map = read_mrc(filename)

    TH_chains = 0.42
    chains_TF = {
        filename:
        read_mrc(filename) > TH_chains
        for filename in chains_list
    }

    for i, hotspot in enumerate(hotspot_list):
        thresholded_map = map > hotspot_list.x284postprocess_threshold_level[i]
        HS_1_footprint_TF_q1 = read_mrc_and_threshold(
            hotspot_list.Bfr1_file_name[i],
            hotspot_list.Bfr1_Molmap_TH[i],
            map.size
        )
        HS_2_footprint_TF_q1 = read_mrc_and_threshold(
            hotspot_list.Bfr2_file_name[i],
            hotspot_list.Bfr2_Molmap_TH[i],
            map.size
        )

        HS_dom_1_TF_q1 = HS_1_footprint_TF_q1 & np.logical_not(HS_2_footprint_TF_q1)
        H2_dom_2_TF_q1 = HS_2_footprint_TF_q1 & np.logical_not(HS_1_footprint_TF_q1)

        for chain in chains_TF:
            ...
            # compute intersected blobs
            map_sum[0] = np.sum(HS_dom_1_q1_of_chain_q2)
            map_sum[1] = np.sum(HS_dom_2_q1_of_chain_q2)
            HS_size_comb = np.sum(
                np.logical_xor(
                    HS_1_footprint_TF_q1,
                    HS_2_footprint_TF_q1
                )
            )
            map_sum[2] = map_sum[1] - map_sum[0]
        
        # compute output sum per size
        


