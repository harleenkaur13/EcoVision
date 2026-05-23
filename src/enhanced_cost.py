# import numpy as np
# import matplotlib.pyplot as plt

# from ndvi import compute_ndvi

# # LOAD IMAGE
# img = np.load("../data/sample/img_0010.npy")

# # LOAD SEGMENTATION
# segmentation = np.load(
#     "../outputs/probability_map.npy"
# )

# # LOAD ROAD PRIOR
# road_prior = np.load(
#     "../outputs/road_prior.npy"
# )

# # NDVI
# ndvi = compute_ndvi(img)

# ndvi_norm = (ndvi + 1) / 2

# # WEIGHTS
# w_ndvi = 0.45
# w_seg = 0.45
# w_road = 0.25

# # ENHANCED COST
# cost_surface = (
#     w_ndvi * ndvi_norm
#     +
#     w_seg * segmentation
#     -
#     w_road * road_prior
# )

# # NORMALIZE
# cost_surface = (
#     cost_surface - cost_surface.min()
# ) / (
#     cost_surface.max() - cost_surface.min()
# )

# # SAVE
# np.save(
#     "../outputs/enhanced_cost.npy",
#     cost_surface
# )

# print("Enhanced cost surface saved!")

# # VISUALIZE
# plt.figure(figsize=(8,8))

# plt.imshow(
#     cost_surface,
#     cmap="inferno"
# )

# plt.colorbar(
#     label="Enhanced Traversal Cost"
# )

# plt.title(
#     "Infrastructure-Aware Cost Surface"
# )

# plt.axis("off")

# plt.show()


import numpy as np

def generate_enhanced_cost(
    rgb,
    segmentation,
    ndvi,
    w_ndvi=0.45,
    w_seg=0.45,
    w_road=0.25
):

    # NORMALIZE NDVI
    ndvi_norm = (ndvi + 1) / 2

    # BRIGHTNESS
    brightness = np.mean(rgb, axis=-1)

    # ROAD PRIOR
    road_prior = (
        0.7 * brightness
        +
        0.3 * (1 - ndvi_norm)
    )

    road_prior = (
        road_prior - road_prior.min()
    ) / (
        road_prior.max() - road_prior.min() + 1e-8
    )

    # ENHANCED COST
    cost = (
        w_ndvi * ndvi_norm
        +
        w_seg * segmentation
        -
        w_road * road_prior
    )

    cost = (
        cost - cost.min()
    ) / (
        cost.max() - cost.min() + 1e-8
    )

    return cost, road_prior