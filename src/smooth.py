def smooth_path(path, step=4):

    if path is None:
        return None

    if len(path) < step:
        return path

    smooth = []

    for i in range(0, len(path), step):

        smooth.append(path[i])

    if smooth[-1] != path[-1]:

        smooth.append(path[-1])

    return smooth