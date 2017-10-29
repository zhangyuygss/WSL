import numpy as np

def process_one(activation, weights_LR, proposal):
    n_feat, w, h = activation.shape
    if proposal is not None:
        activation = activation * proposal
    act_vec = np.reshape(activation, [n_feat, w*h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])
    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec,act_vec)
        heatmap = np.reshape(np.squeeze(heatmap_vec) , [w, h])
        out[:,:,t] = heatmap
    return out


def get_attention_map(activations, weights_LR, proposals):
    # get attention maps for a batch
    _, w, h = activations[0].shape
    n_top = weights_LR.shape[0]
    batch_size = activations.shape[0]
    out = np.zeros([batch_size, w, h, n_top])
    for bs in range(batch_size):
        activation = activations[bs, :, :, :]
        proposal = proposals[bs, :, :] if proposals is not None else None
        out[bs, :, :, :] = process_one(activation, weights_LR, proposal)
    return out