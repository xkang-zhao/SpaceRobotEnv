"""Experimental FP64 CUDA IK for the repository's free-flyer + six-joint chain.

Not used by environments. Pinocchio exports constant model parameters once;
kinematics, inertia coupling, damped solve and SE(3) integration run on CUDA.
"""
import numpy as np
import warp as wp

M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
V6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def chol(a: M6):
    l = M6()
    for i in range(6):
        for j in range(i + 1):
            v = a[i, j]
            for k in range(j):
                v -= l[i, k] * l[j, k]
            if i == j:
                l[i, j] = wp.sqrt(v)
            else:
                l[i, j] = v / l[j, j]
    return l


@wp.func
def chol_solve(l: M6, b: V6):
    y = V6()
    x = V6()
    for i in range(6):
        v = b[i]
        for j in range(i):
            v -= l[i, j] * y[j]
        y[i] = v / l[i, i]
    for t in range(6):
        i = 5 - t
        v = y[i]
        for j in range(i + 1, 6):
            v -= l[j, i] * x[j]
        x[i] = v / l[i, i]
    return x


@wp.kernel
def geometry(q: wp.array2d(dtype=wp.float64),
             placement_p: wp.array(dtype=wp.vec3d), placement_r: wp.array(dtype=wp.quatd),
             axis: wp.array(dtype=wp.vec3d), com: wp.array(dtype=wp.vec3d),
             ee_p: wp.vec3d, ee_r: wp.quatd,
             pos: wp.array2d(dtype=wp.vec3d), rot: wp.array2d(dtype=wp.quatd),
             jv: wp.array3d(dtype=wp.vec3d), jw: wp.array3d(dtype=wp.vec3d),
             jev: wp.array2d(dtype=wp.vec3d), jew: wp.array2d(dtype=wp.vec3d),
             ep: wp.array(dtype=wp.vec3d), er: wp.array(dtype=wp.quatd)):
    w = wp.tid()
    bp = wp.vec3d(q[w, 0], q[w, 1], q[w, 2])
    br = wp.quatd(q[w, 3], q[w, 4], q[w, 5], q[w, 6])
    pos[w, 0] = placement_p[0] + wp.quat_rotate(placement_r[0], bp)
    rot[w, 0] = placement_r[0] * br
    for i in range(1, 7):
        pos[w, i] = pos[w, i-1] + wp.quat_rotate(rot[w, i-1], placement_p[i])
        rot[w, i] = rot[w, i-1] * placement_r[i] * wp.quat_from_axis_angle(axis[i], q[w, i+6])
    ep[w] = pos[w, 6] + wp.quat_rotate(rot[w, 6], ee_p)
    er[w] = rot[w, 6] * ee_r
    for i in range(7):
        center = pos[w, i] + wp.quat_rotate(rot[w, i], com[i])
        for j in range(12):
            linear = wp.vec3d()
            angular = wp.vec3d()
            if j < 6:
                unit = wp.vec3d()
                unit[j % 3] = wp.float64(1.0)
                direction = wp.quat_rotate(rot[w, 0], unit)
                if j < 3:
                    linear = direction
                else:
                    angular = direction
                    linear = wp.cross(direction, center-pos[w, 0])
            elif j - 5 <= i:
                joint = j - 5
                angular = wp.quat_rotate(rot[w, joint], axis[joint])
                linear = wp.cross(angular, center-pos[w, joint])
            jv[w, i, j] = linear
            jw[w, i, j] = angular
    for j in range(12):
        linear = wp.vec3d()
        angular = wp.vec3d()
        if j < 6:
            unit = wp.vec3d()
            unit[j % 3] = wp.float64(1.0)
            direction = wp.quat_rotate(rot[w, 0], unit)
            if j < 3:
                linear = direction
            else:
                angular = direction
                linear = wp.cross(direction, ep[w]-pos[w, 0])
        else:
            joint = j - 5
            angular = wp.quat_rotate(rot[w, joint], axis[joint])
            linear = wp.cross(angular, ep[w]-pos[w, joint])
        jev[w, j] = linear
        jew[w, j] = angular


@wp.kernel
def mass_blocks(rot: wp.array2d(dtype=wp.quatd), jv: wp.array3d(dtype=wp.vec3d),
                jw: wp.array3d(dtype=wp.vec3d), masses: wp.array(dtype=wp.float64),
                inertia: wp.array(dtype=wp.mat33d), mass: wp.array3d(dtype=wp.float64)):
    w, i, j = wp.tid()
    value = wp.float64(0.0)
    for b in range(7):
        wi = wp.quat_rotate_inv(rot[w, b], jw[w, b, i])
        wj = wp.quat_rotate_inv(rot[w, b], jw[w, b, j])
        value += masses[b]*wp.dot(jv[w, b, i], jv[w, b, j]) + wp.dot(wi, inertia[b]*wj)
    mass[w, i, j] = value


@wp.kernel
def residual(ep: wp.array(dtype=wp.vec3d), er: wp.array(dtype=wp.quatd),
             tp: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.quatd),
             errors: wp.array(dtype=V6), norms: wp.array(dtype=wp.float64)):
    w = wp.tid()
    delta = wp.normalize(tr[w] * wp.quat_inverse(er[w]))
    if delta[3] < wp.float64(0.0):
        delta = -delta
    vec = wp.vec3d(delta[0], delta[1], delta[2])
    length = wp.length(vec)
    scale = wp.float64(2.0)
    if length > wp.float64(1.e-12):
        scale = wp.float64(2.0)*wp.atan2(length, delta[3])/length
    angular = vec*scale
    linear = tp[w] - ep[w]
    error = V6(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])
    errors[w] = error
    norms[w] = wp.sqrt(wp.dot(error, error))


@wp.kernel
def update(q: wp.array2d(dtype=wp.float64), mass: wp.array3d(dtype=wp.float64),
           jev: wp.array2d(dtype=wp.vec3d), jew: wp.array2d(dtype=wp.vec3d),
           error: wp.array(dtype=V6), norms: wp.array(dtype=wp.float64),
           velocity: wp.array2d(dtype=wp.float64), damping: wp.float64, dt: wp.float64,
           tolerance: wp.float64):
    w = wp.tid()
    if norms[w] <= tolerance:
        for j in range(12):
            velocity[w, j] = wp.float64(0.0)
        return
    bb = M6()
    bm = M6()
    jb = M6()
    jm = M6()
    for i in range(6):
        for j in range(6):
            bb[i, j] = mass[w, i, j]
            bm[i, j] = mass[w, i, j+6]
            if i < 3:
                jb[i, j] = jev[w, j][i]
                jm[i, j] = jev[w, j+6][i]
            else:
                jb[i, j] = jew[w, j][i-3]
                jm[i, j] = jew[w, j+6][i-3]
    l = chol(bb)
    coupling = M6()
    for j in range(6):
        rhs = V6()
        for i in range(6):
            rhs[i] = bm[i, j]
        solution = chol_solve(l, rhs)
        for i in range(6):
            coupling[i, j] = solution[i]
    jac = jm - jb*coupling
    h = wp.transpose(jac)*jac
    for i in range(6):
        h[i, i] += damping*damping
    arm = chol_solve(chol(h), wp.transpose(jac)*error[w])
    base = -(coupling*arm)
    length = wp.sqrt(wp.dot(base, base) + wp.dot(arm, arm))
    scale = wp.float64(1.0)
    if length > wp.float64(0.2):
        scale = wp.float64(0.2)/length
    arm *= scale
    base *= scale
    for j in range(6):
        velocity[w, j] = base[j]
        velocity[w, j+6] = arm[j]
        q[w, j+7] += dt*arm[j]
    v = wp.vec3d(base[0], base[1], base[2])*dt
    omega = wp.vec3d(base[3], base[4], base[5])*dt
    theta = wp.length(omega)
    a = wp.float64(0.5)
    b = wp.float64(1.0/6.0)
    dq = wp.quatd(omega[0]*wp.float64(0.5), omega[1]*wp.float64(0.5), omega[2]*wp.float64(0.5), wp.float64(1.0))
    if theta > wp.float64(1.e-7):
        a = (wp.float64(1.0)-wp.cos(theta))/(theta*theta)
        b = (theta-wp.sin(theta))/(theta*theta*theta)
        dq = wp.quat_from_axis_angle(omega/theta, theta)
    translated = v + a*wp.cross(omega, v) + b*wp.cross(omega, wp.cross(omega, v))
    br = wp.quatd(q[w, 3], q[w, 4], q[w, 5], q[w, 6])
    translated = wp.quat_rotate(br, translated)
    qr = wp.normalize(br*dq)
    for j in range(3):
        q[w, j] += translated[j]
    for j in range(4):
        q[w, j+3] = qr[j]


class GPUChainIK:
    """Fixed-budget research prototype; unconverged rows must not be applied."""
    def __init__(self, kin, nworld, device='cuda:0'):
        import pinocchio as pin
        m = kin.model
        expected = ['JointModelFreeFlyer','JointModelRZ','JointModelRY','JointModelRY','JointModelRY','JointModelRZ','JointModelRY']
        if nworld < 1 or m.nq != 13 or m.nv != 12 or [j.shortname() for j in list(m.joints)[1:]] != expected or list(m.parents)[1:] != list(range(7)):
            raise ValueError('GPU IK prototype requires the arm.xml free-flyer + six-joint chain')
        frame = m.frames[m.getFrameId(kin.frame_name)]
        if frame.parentJoint != 7:
            raise ValueError('GPU IK prototype requires an EE frame on joint 7')
        wp.init()
        self.device, self.nworld = device, nworld
        def array(values, dtype):
            return wp.array(np.asarray(values), dtype=dtype, device=device)
        self.pp = array([p.translation for p in list(m.jointPlacements)[1:]], wp.vec3d)
        self.pr = array([pin.Quaternion(p.rotation).coeffs() for p in list(m.jointPlacements)[1:]], wp.quatd)
        self.axis = array([[0,0,0],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]], wp.vec3d)
        self.com = array([i.lever for i in list(m.inertias)[1:]], wp.vec3d)
        self.masses = array([i.mass for i in list(m.inertias)[1:]], wp.float64)
        self.inertia = array([i.inertia for i in list(m.inertias)[1:]], wp.mat33d)
        self.ee_p = wp.vec3d(*frame.placement.translation)
        self.ee_r = wp.quatd(*pin.Quaternion(frame.placement.rotation).coeffs())
        def zeros(shape, dtype):
            return wp.zeros(shape, dtype=dtype, device=device)
        self.q = zeros((nworld,13),wp.float64)
        self.tp, self.tr = zeros(nworld,wp.vec3d), zeros(nworld,wp.quatd)
        self.pos, self.rot = zeros((nworld,7),wp.vec3d), zeros((nworld,7),wp.quatd)
        self.jv, self.jw = zeros((nworld,7,12),wp.vec3d), zeros((nworld,7,12),wp.vec3d)
        self.jev, self.jew = zeros((nworld,12),wp.vec3d), zeros((nworld,12),wp.vec3d)
        self.ep, self.er = zeros(nworld,wp.vec3d), zeros(nworld,wp.quatd)
        self.mass = zeros((nworld,6,12),wp.float64)
        self.errors, self.norms = zeros(nworld,V6), zeros(nworld,wp.float64)
        self.velocity = zeros((nworld,12),wp.float64)
        self.graphs = {}

    def upload(self, q, targets):
        import pinocchio as pin
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.nworld,13) or len(targets) != self.nworld or not np.isfinite(q).all():
            raise ValueError('Expected finite q (nworld,13) and one target per world')
        if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-10,rtol=0):
            raise ValueError('Base quaternion must be normalized, xyzw')
        positions = np.array([p.translation for p in targets])
        rotations = np.array([pin.Quaternion(p.rotation).coeffs() for p in targets])
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError('GPU IK targets must be finite')
        self.q.assign(q)
        self.tp.assign(positions)
        self.tr.assign(rotations)

    def evaluate(self):
        n=self.nworld
        wp.launch(geometry,n,inputs=[self.q,self.pp,self.pr,self.axis,self.com,self.ee_p,self.ee_r,self.pos,self.rot,self.jv,self.jw,self.jev,self.jew,self.ep,self.er],device=self.device)
        wp.launch(mass_blocks,(n,6,12),inputs=[self.rot,self.jv,self.jw,self.masses,self.inertia,self.mass],device=self.device)
        wp.launch(residual,n,inputs=[self.ep,self.er,self.tp,self.tr,self.errors,self.norms],device=self.device)

    def step(self, dt=0.2, damping=0.05, tolerance=-1.0):
        self.evaluate()
        wp.launch(update,self.nworld,inputs=[self.q,self.mass,self.jev,self.jew,self.errors,self.norms,self.velocity,damping,dt,tolerance],device=self.device)

    def prepare(self, iterations=40, dt=0.2, damping=0.05, tolerance=1e-4):
        """Warm/capture kernels; upload inputs again before launching the graph."""
        if iterations < 1 or dt <= 0 or damping <= 0:
            raise ValueError('iterations, dt and damping must be positive')
        key=(iterations,dt,damping,tolerance)
        if key not in self.graphs:
            self.step(dt,damping,tolerance)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(iterations):
                    self.step(dt,damping,tolerance)
                self.evaluate()
            self.graphs[key]=capture.graph
        return self.graphs[key]

    def read(self, tolerance=1e-4):
        q, residuals = self.q.numpy(), self.norms.numpy()
        return q, residuals, np.isfinite(q).all(axis=1) & np.isfinite(residuals) & (residuals <= tolerance)
