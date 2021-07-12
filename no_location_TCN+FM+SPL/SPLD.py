import numpy as np


# spld
def spld(loss, group_member_ship, lam, gamma):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        # print(idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        # print(sorted_idx_in_group_arr)

        for (i, ii) in enumerate(sorted_idx_in_group_arr):
            if loss[ii] < (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i))):
                selected_idx.append(ii)
            else:
                pass
            selected_score[ii] = loss[ii] - (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i)))

    selected_idx_arr = np.array(selected_idx)
    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]

    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])

    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    return sorted_idx_in_selected_samples_arr
