"""GPU state observations and grasp confirmation for SpaceUR10e worlds."""

import warp as wp


@wp.kernel
def mark_contacts(nacon: wp.array[int], worldid: wp.array[int], geom: wp.array[wp.vec2i],
                  roles: wp.array[int], bits: wp.array[int]):
    i = wp.tid()
    if i < nacon[0]:
        w = worldid[i]
        if w >= 0 and w < bits.shape[0]:
            a, b = geom[i][0], geom[i][1]
            if a >= 0 and b >= 0:
                ra, rb = roles[a], roles[b]
                if ra & 4:
                    wp.atomic_or(bits, w, rb & 3)
                elif rb & 4:
                    wp.atomic_or(bits, w, ra & 3)


@wp.kernel
def clear_counts(mask: wp.array[bool], counts: wp.array[int]):
    w = wp.tid()
    if mask[w]:
        counts[w] = 0


@wp.kernel
def masked_rows(mask: wp.array[bool], source: wp.array2d[float], dest: wp.array2d[float]):
    w, j = wp.tid()
    if mask[w]:
        dest[w, j] = source[w, j]


@wp.kernel
def masked_scalar(mask: wp.array[bool], source: wp.array[float], dest: wp.array[float]):
    w = wp.tid()
    if mask[w]:
        dest[w] = source[w]


@wp.kernel
def masked_counts(mask: wp.array[bool], source: wp.array[int], dest: wp.array[int]):
    w = wp.tid()
    if mask[w]:
        dest[w] = source[w]


@wp.kernel
def pack_state(qpos: wp.array2d[float], qvel: wp.array2d[float], ctrl: wp.array2d[float],
               sites: wp.array2d[wp.vec3], matrices: wp.array2d[wp.mat33],
               bodies: wp.array2d[wp.vec3], quats: wp.array2d[wp.quat],
               actions: wp.array2d[float], bits: wp.array[int], counts: wp.array[int],
               ids: wp.array[int], params: wp.array[float], update: int,
               obs: wp.array2d[float], teacher: wp.array2d[float], metrics: wp.array2d[float],
               terminated: wp.array[int], truncated: wp.array[int]):
    w = wp.tid()
    ee, pinch, target, grip, dof = ids[0], ids[1], ids[2], ids[3], ids[4]
    for j in range(6):
        obs[w, j] = qpos[w, 7 + j]
    for j in range(7):
        obs[w, 6 + j] = qpos[w, j]
    quat = wp.quat_from_matrix(matrices[w, ee])  # Warp quaternion is xyzw.
    for j in range(3):
        obs[w, 13 + j] = sites[w, ee][j]
        teacher[w, j] = bodies[w, target][j]
        teacher[w, 7 + j] = sites[w, pinch][j]
    obs[w, 16] = quat[3]
    obs[w, 17] = quat[0]
    obs[w, 18] = quat[1]
    obs[w, 19] = quat[2]
    for j in range(4):
        teacher[w, 3 + j] = quats[w, target][j]  # MJWarp body quaternions are wxyz.
    obs[w, 20] = wp.clamp((qpos[w, grip] - params[0]) / (params[1] - params[0]), 0.0, 1.0)
    speed_sq = 0.0
    for j in range(6):
        speed_sq += qvel[w, dof + j] * qvel[w, dof + j]
    success = bits[w] == 3 and ctrl[w, 6] > params[2] and wp.sqrt(speed_sq) < params[3]
    if update != 0:
        if success:
            counts[w] += 1
        else:
            counts[w] = 0
    terminated[w] = int(counts[w] >= ids[5])
    dist = wp.length(sites[w, pinch] - bodies[w, target])
    truncated[w] = int(dist > params[4])
    reach = -params[6] * dist
    align = 0.0
    if dist < params[5]:
        dot = 0.0
        for j in range(4):
            dot += obs[w, 16 + j] * teacher[w, 3 + j]
        align = -params[7] * 2.0 * wp.acos(wp.clamp(wp.abs(dot), 0.0, 1.0))
    contact = 0.0
    if bits[w] != 0:
        contact += params[8]
    if bits[w] == 3:
        contact += params[9]
    success_reward = 0.0
    if success:
        success_reward = params[10]
    action_sq = 0.0
    for j in range(7):
        action_sq += actions[w, j] * actions[w, j]
    action_penalty = -params[11] * action_sq
    metrics[w, 0] = dist
    metrics[w, 1] = reach
    metrics[w, 2] = align
    metrics[w, 3] = contact
    metrics[w, 4] = success_reward
    metrics[w, 5] = action_penalty
    metrics[w, 6] = reach + align + contact + success_reward + action_penalty
    metrics[w, 7] = float((bits[w] & 1) != 0)
    metrics[w, 8] = float((bits[w] & 2) != 0)
