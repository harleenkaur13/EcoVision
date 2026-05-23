# import heapq
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# os.makedirs("../outputs", exist_ok=True)

# # LOAD ENHANCED COST SURFACE
# cost_surface = np.load(
#     "../outputs/enhanced_cost.npy"
# )

# HEIGHT, WIDTH = cost_surface.shape

# # INTERACTIVE POINTS
# clicked_points = []

# def onclick(event):

#     if event.xdata is None or event.ydata is None:
#         return

#     x = int(event.xdata)
#     y = int(event.ydata)

#     clicked_points.append((y, x))

#     plt.scatter(
#         x,
#         y,
#         color="cyan",
#         s=100
#     )

#     plt.draw()

#     if len(clicked_points) == 2:
#         plt.close()

# # DISPLAY
# fig = plt.figure(figsize=(8,8))

# plt.imshow(
#     cost_surface,
#     cmap="inferno"
# )

# plt.title(
#     "Click START then GOAL"
# )

# fig.canvas.mpl_connect(
#     "button_press_event",
#     onclick
# )

# plt.show()

# start = clicked_points[0]
# goal = clicked_points[1]

# print("Start:", start)
# print("Goal:", goal)

# # DIRECTIONS
# directions = [
#     (-1,0),
#     (1,0),
#     (0,-1),
#     (0,1),
#     (-1,-1),
#     (-1,1),
#     (1,-1),
#     (1,1)
# ]

# # HEURISTIC
# def heuristic(a, b):

#     return np.sqrt(
#         (a[0]-b[0])**2 +
#         (a[1]-b[1])**2
#     )

# # ENHANCED A*
# def astar(cost_map, start, goal):

#     open_set = []

#     heapq.heappush(
#         open_set,
#         (0, start)
#     )

#     came_from = {}

#     g_score = {
#         start: 0
#     }

#     f_score = {
#         start: heuristic(start, goal)
#     }

#     while open_set:

#         current = heapq.heappop(open_set)[1]

#         if current == goal:

#             path = []

#             while current in came_from:

#                 path.append(current)

#                 current = came_from[current]

#             path.append(start)

#             return path[::-1]

#         for dx, dy in directions:

#             nx = current[0] + dx
#             ny = current[1] + dy

#             neighbor = (nx, ny)

#             # BOUNDS
#             if (
#                 nx < 0 or
#                 ny < 0 or
#                 nx >= HEIGHT or
#                 ny >= WIDTH
#             ):
#                 continue

#             # ENHANCED TERRAIN COST
#             terrain_cost = (
#                 cost_map[nx, ny] * 8
#             )

#             # MOVEMENT COST
#             movement_cost = (
#                 1.4 if dx != 0 and dy != 0
#                 else 1.0
#             )

#             # SLIGHT MOMENTUM BONUS
#             momentum_bonus = 0.0

#             tentative_g = (
#                 g_score[current]
#                 +
#                 terrain_cost
#                 +
#                 movement_cost
#                 +
#                 momentum_bonus
#             )

#             if (
#                 neighbor not in g_score
#                 or
#                 tentative_g < g_score[neighbor]
#             ):

#                 came_from[neighbor] = current

#                 g_score[neighbor] = tentative_g

#                 # REDUCED HEURISTIC STRENGTH
#                 heuristic_weight = 0.7

#                 f_score[neighbor] = (
#                     tentative_g
#                     +
#                     heuristic_weight *
#                     heuristic(neighbor, goal)
#                 )

#                 heapq.heappush(
#                     open_set,
#                     (
#                         f_score[neighbor],
#                         neighbor
#                     )
#                 )

#     return None

# # RUN
# print("Running enhanced A*...")

# path = astar(
#     cost_surface,
#     start,
#     goal
# )

# print("Finished!")

# if path is None:

#     print("No valid path found!")

# else:

#     print("Path length:", len(path))

#     np.save(
#         "../outputs/enhanced_path.npy",
#         np.array(path)
#     )

#     print("Enhanced path saved!")

# # VISUALIZATION
# plt.figure(figsize=(8,8))

# plt.imshow(
#     cost_surface,
#     cmap="inferno"
# )

# if path is not None:

#     path_x = [p[1] for p in path]
#     path_y = [p[0] for p in path]

#     plt.plot(
#         path_x,
#         path_y,
#         color="cyan",
#         linewidth=3
#     )

# plt.scatter(
#     start[1],
#     start[0],
#     color="lime",
#     s=120,
#     edgecolors="black",
#     label="Start"
# )

# plt.scatter(
#     goal[1],
#     goal[0],
#     color="red",
#     s=120,
#     edgecolors="black",
#     label="Goal"
# )

# plt.legend()

# plt.title(
#     "Infrastructure-Aware Eco Routing"
# )

# plt.axis("off")

# plt.show()


import heapq
import numpy as np

directions = [
    (-1,0),
    (1,0),
    (0,-1),
    (0,1),
    (-1,-1),
    (-1,1),
    (1,-1),
    (1,1)
]

def heuristic(a, b):

    return np.sqrt(
        (a[0]-b[0])**2 +
        (a[1]-b[1])**2
    )

def astar(cost_map, start, goal):

    HEIGHT, WIDTH = cost_map.shape

    open_set = []

    heapq.heappush(
        open_set,
        (0, start)
    )

    came_from = {}

    g_score = {
        start: 0
    }

    f_score = {
        start: heuristic(start, goal)
    }

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current == goal:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[current]

            path.append(start)

            return path[::-1]

        for dx, dy in directions:

            nx = current[0] + dx
            ny = current[1] + dy

            neighbor = (nx, ny)

            if (
                nx < 0 or
                ny < 0 or
                nx >= HEIGHT or
                ny >= WIDTH
            ):
                continue

            terrain_cost = (
                cost_map[nx, ny] * 8
            )

            movement_cost = (
                1.4 if dx != 0 and dy != 0
                else 1.0
            )

            tentative_g = (
                g_score[current]
                +
                terrain_cost
                +
                movement_cost
            )

            if (
                neighbor not in g_score
                or
                tentative_g < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                heuristic_weight = 0.7

                f_score[neighbor] = (
                    tentative_g
                    +
                    heuristic_weight *
                    heuristic(neighbor, goal)
                )

                heapq.heappush(
                    open_set,
                    (
                        f_score[neighbor],
                        neighbor
                    )
                )

    return None