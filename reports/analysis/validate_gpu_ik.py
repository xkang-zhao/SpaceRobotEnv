"""Validate experimental CUDA IK against Pinocchio and time fixed-input batches."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import pinocchio as pin
import warp as wp
from mujoco_robot.robot_ik import Kinematics
from mujoco_robot.gpu_ik_prototype import GPUChainIK


def fixtures(kin):
    """Independent configurations, translated/rotated bases, mixed targets."""
    rng=np.random.default_rng(42)
    qs=[];targets=[];kinds=[]
    frame=kin.model.getFrameId(kin.frame_name)
    for i in range(96):
        q=pin.integrate(kin.model,pin.neutral(kin.model),rng.normal(0,.4,kin.model.nv))
        if i==0:
            q=pin.neutral(kin.model)
        pin.framesForwardKinematics(kin.model,kin.data,q)
        pose=pin.SE3(kin.data.oMf[frame])
        dp=rng.normal(0,.002,3);dr=rng.normal(0,.003,3)
        kind='small'
        if i%12==0:
            dp=np.zeros(3);dr=np.zeros(3);kind='aligned'
        elif i%12==1:
            dp=rng.normal(0,.3,3);dr=rng.normal(0,.4,3);kind='large'
        elif i%12==2:
            dr=np.array([np.pi-1e-7,0.,0.]);kind='near_pi'
        elif i%12==3:
            dp=np.array([100.,100.,100.]);kind='unreachable'
        qs.append(q);targets.append(pin.SE3(pose.rotation@pin.exp3(dr),pose.translation+dp));kinds.append(kind)
    return np.array(qs),targets,kinds


def record_cube():
    import gymnasium as gym
    import envs
    from planners.auto_grasp.tasks import AutoGraspSession
    env=gym.make('SpaceUR10e-Cube-v0',observation_mode='state',render_mode=None)
    rows=[];targets=[]
    kin=env.unwrapped.kinematics;old=kin.ik
    def record(q,target,*args,**kwargs):
        rows.append(q.copy());targets.append(pin.SE3(target))
        return old(q,target,*args,**kwargs)
    try:
        session=AutoGraspSession('cube',env);obs=session.reset(7)
        kin.ik=record
        with redirect_stdout(io.StringIO()):
            for _ in range(session.max_steps):
                action=session.compute_action({**obs,'target_pose':env.unwrapped._get_target_pose()})
                tr=env.step(action);session.update(*tr);obs=tr[0]
                if session.is_done:break
        if not (tr[2] and tr[4].get('is_success')):
            raise RuntimeError('CPU Cube seed 7 failed while recording IK fixtures')
        return np.array(rows),targets
    finally:
        env.close()


def cpu_fixed(kin,qs,targets):
    frame=kin.model.getFrameId(kin.frame_name)
    outputs=[]
    for q,target in zip(qs,targets):
        q=q.copy()
        for _ in range(40):
            if kin._target_error(q,target,frame)<=1e-4:break
            q,_=kin.step_ik(q,target,frame,dt=.2,damping=.05)
        outputs.append(q)
    return np.array(outputs)


def validate(kin):
    qs,targets,kinds=fixtures(kin);frame=kin.model.getFrameId(kin.frame_name)
    gpu=GPUChainIK(kin,len(qs));gpu.upload(qs,targets);gpu.evaluate()
    mass=gpu.mass.numpy();jv=gpu.jev.numpy();jw=gpu.jew.numpy()
    positions=gpu.ep.numpy();rotations=gpu.er.numpy();errors=gpu.errors.numpy()
    maxima=dict(mass_abs=0.,jacobian_abs=0.,pose_abs=0.,error_vector_abs=0.)
    for i,(q,target) in enumerate(zip(qs,targets)):
        pin.crba(kin.model,kin.data,q)
        maxima['mass_abs']=max(maxima['mass_abs'],float(np.max(abs(mass[i]-kin.data.M[:6]))))
        pin.computeJointJacobians(kin.model,kin.data,q);pin.updateFramePlacements(kin.model,kin.data)
        jac=pin.getFrameJacobian(kin.model,kin.data,frame,pin.LOCAL_WORLD_ALIGNED)
        maxima['jacobian_abs']=max(maxima['jacobian_abs'],float(np.max(abs(np.r_[jv[i].T,jw[i].T]-jac))))
        pose=kin.data.oMf[frame]
        maxima['pose_abs']=max(maxima['pose_abs'],float(np.max(abs(positions[i]-pose.translation))),float(np.max(abs(pin.Quaternion(rotations[i]).matrix()-pose.rotation))))
        err=np.r_[target.translation-pose.translation,pose.rotation@pin.log3(pose.rotation.T@target.rotation)]
        maxima['error_vector_abs']=max(maxima['error_vector_abs'],float(np.max(abs(errors[i]-err))))
    expected=np.array([kin.step_ik(q,t,frame,dt=.2,damping=.05)[0] for q,t in zip(qs,targets)])
    gpu.step();actual=gpu.q.numpy();velocity=gpu.velocity.numpy()
    maxima['single_step_q_abs']=float(np.max(abs(actual-expected)))
    numerator=np.linalg.norm(np.einsum('nij,nj->ni',mass,velocity),axis=1)
    denominator=np.linalg.norm(np.einsum('nij,nj->ni',mass[:,:,:6],velocity[:,:6]),axis=1)+np.linalg.norm(np.einsum('nij,nj->ni',mass[:,:,6:],velocity[:,6:]),axis=1)
    maxima['momentum_relative']=float(np.max(numerator/np.maximum(denominator,1e-15)))
    graph=gpu.prepare();gpu.upload(qs,targets);wp.capture_launch(graph);wp.synchronize_device(gpu.device)
    actual,norms,accepted=gpu.read();expected=cpu_fixed(kin,qs,targets)
    maxima['forty_step_q_abs']=float(np.max(abs(actual-expected)))
    verified=np.array([kin._target_error(q,t,frame) for q,t in zip(actual,targets)])
    maxima['final_residual_abs']=float(np.max(abs(verified-norms)))
    assert np.isfinite(actual).all() and np.isfinite(norms).all()
    for name,value in maxima.items():
        assert value<1e-7,(name,value)
    assert np.array_equal(accepted,verified<=1e-4)
    groups={kind:dict(total=kinds.count(kind),accepted=int(sum(a for a,k in zip(accepted,kinds) if k==kind))) for kind in sorted(set(kinds))}
    assert groups['unreachable']['accepted']==0
    return dict(samples=len(qs),max_errors=maxima,groups=groups)


def benchmark(kin,qs,targets,n):
    indices=np.arange(n)%len(qs);q=qs[indices];t=[targets[i] for i in indices]
    start=perf_counter();gpu=GPUChainIK(kin,n);gpu.upload(q,t);graph=gpu.prepare();wp.synchronize_device(gpu.device)
    setup=perf_counter()-start
    timings=[]
    for _ in range(3):
        start=perf_counter();gpu.upload(q,t);wp.synchronize_device(gpu.device);uploaded=perf_counter()
        wp.capture_launch(graph);wp.synchronize_device(gpu.device);solved=perf_counter()
        actual,residual,accepted=gpu.read();wp.synchronize_device(gpu.device);read=perf_counter()
        timings.append([(uploaded-start)*1000,(solved-uploaded)*1000,(read-solved)*1000])
    start=perf_counter();expected=cpu_fixed(kin,q,t);fixed_ms=(perf_counter()-start)*1000
    start=perf_counter()
    with redirect_stdout(io.StringIO()):
        production=np.array([kin.ik(x,y,steps=20)[0] for x,y in zip(q,t)])
    production_ms=(perf_counter()-start)*1000
    diff=float(np.max(abs(expected-actual)));assert diff<1e-7,diff
    frame=kin.model.getFrameId(kin.frame_name)
    verified=np.array([kin._target_error(x,y,frame) for x,y in zip(actual,t)])
    assert np.isfinite(actual).all() and np.array_equal(accepted,verified<=1e-4)
    mean=np.mean(timings,axis=0)
    position_errors=[];rotation_errors=[]
    for x,y in zip(actual,t):
        pin.framesForwardKinematics(kin.model,kin.data,x)
        pose=kin.data.oMf[frame]
        position_errors.append(float(np.linalg.norm(pose.translation-y.translation)))
        rotation_errors.append(float(np.linalg.norm(pin.log3(pose.rotation.T@y.rotation))))
    base_angles=[float(np.linalg.norm(pin.log3(pin.Quaternion(x[3:7]).matrix().T@pin.Quaternion(y[3:7]).matrix()))) for x,y in zip(actual,production)]
    return dict(worlds=n,setup_s=setup,cpu_fixed_ms=fixed_ms,cpu_production_ms=production_ms,
                gpu_upload_ms=mean[0],gpu_compute_ms=mean[1],gpu_readback_ms=mean[2],gpu_total_ms=float(sum(mean)),
                gpu_timing_samples_ms=timings,accepted=int(accepted.sum()),needs_fallback=int((~accepted).sum()),
                q_abs_vs_same_schedule=diff,q_abs_vs_production=float(np.max(abs(actual-production))),
                max_joint_difference_vs_production_rad=float(np.max(abs(actual[:,7:]-production[:,7:]))),
                max_base_translation_difference_vs_production_m=float(np.max(np.linalg.norm(actual[:,:3]-production[:,:3],axis=1))),
                max_base_rotation_difference_vs_production_rad=max(base_angles),
                max_ee_position_error_m=max(position_errors),max_ee_rotation_error_rad=max(rotation_errors),
                max_accepted_residual=float(np.max(verified[accepted])) if accepted.any() else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds',nargs='+',type=int,default=[1,32,128,512])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    kin=Kinematics('mjcf/arm.xml','left_attachment')
    checks=validate(kin);print('VALIDATION',json.dumps(checks),flush=True)
    q,t=record_cube();rows=[]
    for n in args.worlds:
        row=benchmark(kin,q,t,n);rows.append(row);print('BENCHMARK',json.dumps(row),flush=True)
    with args.output.open('x') as f:
        json.dump(dict(validation=checks,recorded_cube_states=len(q),benchmark=rows,notes='FP64 isolated prototype, full GPU dynamics and IK. Fixed 40-iteration budget, 1e-4 stop. No trajectory fallback on GPU; unaccepted rows require CPU fallback. Timing includes synchronous upload/readback separately. CPU serial baseline. No environment integration or grasp closed-loop validation.'),f,indent=2)


if __name__=='__main__':main()
