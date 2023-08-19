# import matplotlib.pyplot as plt


# x_initial = [0, 667, 1334, 2000, 667, 1334, 0, 667, 1334, 2000] # 3*D, 6 * D, 6 * D,
# y_initial = [0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0, 2000.0] # 4 * D, 0, 4 * D,
# x_opt = [100.0]
# y_opt = [100.0]

# fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# fontsize = 12
# ax.plot(x_initial, y_initial, "ob")
# ax.plot(x_opt, y_opt, "or")

# ax.set_xlabel("x (m)", fontsize=fontsize)
# ax.set_ylabel("y (m)", fontsize=fontsize)
# ax.axis("equal")
# ax.grid()
# ax.tick_params(which="both", labelsize=fontsize)

# # # Create legend for the first group of labels
# legend1 = ax.legend(
#     ["Old locations", "New locations"],
#     # loc="upper right",
#     bbox_to_anchor=(0.5, 1.01),
#     ncol=2,
#     fontsize=fontsize,
# )

# # Add the first legend to the plot
# ax.add_artist(legend1)
# length = 2000.0
# boundaries = [(0.0, 0.0), (0.0, length), (length, length), (length, 0.0), (0.0, 0.0)]
# verts = boundaries
# for i in range(len(verts)):
#     if i == len(verts) - 1:
#         plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
#     else:
#         plt.plot(
#             [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
#         )

# # Create legend for the second group of labels
# # plot_boundary(ax, ["fixed", "flexible"], ["--", ":"])
# boundary_styles = ["--", ":"]
# boundary_names = ["fixed", "flexible"]
# boundary_list = [[(0.0, 0.0), (0.0, 100), (100, 100), (100, 0.0), (0.0, 0.0)],\
#                     [(0.0, 0.0), (0.0, 200), (200, 200), (200, 0.0), (0.0, 0.0)]]
# for i in range(len(boundary_list)):
#     x_coords, y_coords = zip(*boundary_list[i])
#     ax.plot(x_coords, y_coords, linestyle=boundary_styles[i], label=boundary_names[i], color='k')
#     legend2 = ax.legend()

# ax.add_artist(legend2)


# # Save the plot
# # plt.savefig(path)
# plt.show()
import numpy as np
variance_wd_range = np.linspace(15, 90, 6)#4
print(variance_wd_range)