"""Race-free replacements for SpaceUR10e smooth-tree reductions and mass rows."""

import warp as wp
from mujoco_warp._src import math
from mujoco_warp._src.types import vec10


@wp.kernel
def accumulate_subtree_com_serial(
    body_parentid: wp.array[int],
    subtree_com: wp.array2d[wp.vec3],
):
    """Accumulate children in stable reverse body order, one thread per world."""
    worldid = wp.tid()
    bodyid = body_parentid.shape[0] - 1
    while bodyid > 0:
        parentid = body_parentid[bodyid]
        subtree_com[worldid, parentid] = (
            subtree_com[worldid, parentid] + subtree_com[worldid, bodyid]
        )
        bodyid -= 1


@wp.kernel
def accumulate_crb_serial(
    body_parentid: wp.array[int],
    crb: wp.array2d[vec10],
):
    """Accumulate composite inertias without cross-thread floating atomics."""
    worldid = wp.tid()
    bodyid = body_parentid.shape[0] - 1
    while bodyid > 0:
        parentid = body_parentid[bodyid]
        if parentid != 0:
            crb[worldid, parentid] = crb[worldid, parentid] + crb[worldid, bodyid]
        bodyid -= 1


@wp.kernel
def accumulate_cfrc_serial(
    body_parentid: wp.array[int],
    cfrc: wp.array2d[wp.spatial_vector],
):
    """Propagate internal forces in stable reverse body order."""
    worldid = wp.tid()
    bodyid = body_parentid.shape[0] - 1
    while bodyid > 0:
        parentid = body_parentid[bodyid]
        cfrc[worldid, parentid] = cfrc[worldid, parentid] + cfrc[worldid, bodyid]
        bodyid -= 1


@wp.kernel
def mass_matrix_rows(
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    dof_armature: wp.array2d[float],
    matrix_rownnz: wp.array[int],
    matrix_rowadr: wp.array[int],
    cdof: wp.array2d[wp.spatial_vector],
    crb: wp.array2d[vec10],
    matrix: wp.array2d[float],
):
    """Build one disjoint packed mass-matrix row per thread."""
    worldid, dofid = wp.tid()
    bodyid = dof_bodyid[dofid]
    matrix_index = matrix_rowadr[dofid] + matrix_rownnz[dofid] - 1
    matrix[worldid, matrix_index] = dof_armature[
        worldid % dof_armature.shape[0], dofid
    ]
    buffer = math.inert_vec(crb[worldid, bodyid], cdof[worldid, dofid])
    while dofid >= 0:
        matrix[worldid, matrix_index] = (
            matrix[worldid, matrix_index] + wp.dot(cdof[worldid, dofid], buffer)
        )
        matrix_index -= 1
        dofid = dof_parentid[dofid]


def install(smooth):
    """Replace MuJoCo Warp smooth reductions while reusing its race-free kernels."""

    def com_pos(model, data):
        wp.launch(
            smooth._subtree_com_init,
            dim=(data.nworld, model.nbody),
            inputs=[model.body_mass, data.xipos],
            outputs=[data.subtree_com],
        )
        wp.launch(
            accumulate_subtree_com_serial,
            dim=data.nworld,
            inputs=[model.body_parentid, data.subtree_com],
        )
        wp.launch(
            smooth._subtree_div,
            dim=(data.nworld, model.nbody),
            inputs=[model.body_subtreemass, data.subtree_com],
            outputs=[data.subtree_com],
        )
        wp.launch(
            smooth._cinert,
            dim=(data.nworld, model.nbody),
            inputs=[
                model.body_rootid,
                model.body_mass,
                model.body_inertia,
                data.xipos,
                data.ximat,
                data.subtree_com,
            ],
            outputs=[data.cinert],
        )
        wp.launch(
            smooth._cdof,
            dim=(data.nworld, model.njnt),
            inputs=[
                model.body_rootid,
                model.jnt_type,
                model.jnt_dofadr,
                model.jnt_bodyid,
                data.xmat,
                data.xanchor,
                data.xaxis,
                data.subtree_com,
            ],
            outputs=[data.cdof],
        )

    def crb(model, data):
        wp.copy(data.crb, data.cinert)
        wp.launch(
            accumulate_crb_serial,
            dim=data.nworld,
            inputs=[model.body_parentid, data.crb],
        )
        data.M.zero_()
        wp.launch(
            mass_matrix_rows,
            dim=(data.nworld, model.nv),
            inputs=[
                model.dof_bodyid,
                model.dof_parentid,
                model.dof_armature,
                model.M_rownnz,
                model.M_rowadr,
                data.cdof,
                data.crb,
            ],
            outputs=[data.M],
        )

    def rne_cfrc_backward(model, data):
        wp.launch(
            accumulate_cfrc_serial,
            dim=data.nworld,
            inputs=[model.body_parentid, data.cfrc_int],
        )

    smooth.com_pos = com_pos
    smooth.crb = crb
    smooth._rne_cfrc_backward = rne_cfrc_backward


def install_ccd_record_capacity(collision_convex):
    """Give dynamic CCD counter atomics enough deterministic record slots."""
    original = collision_convex.ccd_kernel_builder
    if getattr(original, "_spaceur10e_deterministic", False):
        return
    configured_modules = set()

    def ccd_kernel_builder(*args, **kwargs):
        kernel = original(*args, **kwargs)
        module_id = id(kernel.module)
        if module_id not in configured_modules:
            wp.set_module_options(
                {"deterministic_max_records": 16}, module=kernel.module
            )
            configured_modules.add(module_id)
        return kernel

    ccd_kernel_builder._spaceur10e_deterministic = True
    collision_convex.ccd_kernel_builder = ccd_kernel_builder
