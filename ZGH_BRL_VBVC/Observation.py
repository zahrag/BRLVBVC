
import numpy as np
from ZGH_BRL_VBVC.helpers import normalize


def get_scene_features(seg_image, normalize_input, margin_percentage, f_sz=6, segment_image=False):

    # Semantic Segmentation Feature Extraction
    # seg_dim = seg_image.data.shape
    cnt_lable = 13  # Number of semantic features of the segmented image
    image_features = np.zeros((2, cnt_lable))
    image_features[0, :] = range(cnt_lable)

    img_sz = seg_image.shape
    if segment_image:
        class_weights = [1, 20, 1, 1, 1]
    else:
        class_weights = [1, 20, 1, 1, 1]

    state_feature = np.zeros((f_sz*6, 1))
    his = 0
    for margin in margin_percentage:
        marg = [0, 0, 0, 0]
        marg[0] = img_sz[0] * margin[0]
        marg[1] = img_sz[0] * margin[1]
        marg[2] = img_sz[1] * margin[2]
        marg[3] = img_sz[1] * margin[3]

        hist_mat = np.histogram(seg_image[int(marg[0]):int(marg[1] + 1), int(marg[2]):int(marg[3] + 1)], bins=cnt_lable, range=(0,13))
        image_features[1, :] = hist_mat[0]

        state_feature[his * f_sz + 0, 0] = image_features[1, 7] * class_weights[0]  # Roads
        state_feature[his * f_sz + 1, 0] = image_features[1, 6] * class_weights[1]  # RoadLines
        state_feature[his * f_sz + 2, 0] = (image_features[1, 8] + image_features[1, 3] + image_features[1, 0]) * class_weights[2]  # SideWalks + Other + None (Outside of road)
        state_feature[his * f_sz + 3, 0] = (image_features[1, 4] + image_features[1, 10]) * class_weights[3]  # Dynamic Obstacles
        state_feature[his * f_sz + 4, 0] = (image_features[1, 1] + image_features[1, 2] + image_features[1, 5] +\
                                    image_features[1, 9] + image_features[1, 11] + image_features[1, 12]) * class_weights[4]  # Static Obstacles
        his += 1

    if normalize_input:
        state_feature = normalize(state_feature)

    return state_feature

