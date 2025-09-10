import time
import numpy as np
import mujoco

try:
    import mujoco.viewer as viewer
except Exception as e:
    raise SystemExit(f"Could not import mujoco.viewer: {e}")

# Load model directly from file path
model = mujoco.MjModel.from_xml_path('pendulum.xml')
data = mujoco.MjData(model)

print(f"MuJoCo version: {getattr(mujoco, '__version__', 'unknown')}")
print(f"nq (qpos size): {model.nq}, nv (qvel size): {model.nv}")

# Choose a DoF index to manipulate. 0..model.nv-1
# For this XML: first joint is a ball joint (3 velocity DOFs), then hinges.
# We'll pick the first hinge after the ball: index 3 if model.nv > 3 else 0.
vel_dof = 3 if model.nv > 3 else 0
print(f"Controlling velocity DOF index: {vel_dof}")

AMPL = 2.0
FREQ = 0.5  # Hz
DURATION = 10.0

if hasattr(viewer, 'launch_passive'):
    with viewer.launch_passive(model, data) as v:
        start_t = data.time
        while v.is_running() and (data.time - start_t) < DURATION:
            # Directly write desired velocity (demonstration only)
            data.qvel[vel_dof] = AMPL * np.sin(2 * np.pi * FREQ * data.time)
            mujoco.mj_step(model, data)
            if int(data.time / model.opt.timestep) % 400 == 0:
                print(f"t={data.time:.3f} qpos0={data.qpos[0]:.3f} qvel[{vel_dof}]={data.qvel[vel_dof]:.3f}")
            v.sync()
            time.sleep(0.001)
else:
    print("Passive viewer not available; using blocking viewer after pre-sim.")
    end_time = time.time() + DURATION
    while time.time() < end_time:
        data.qvel[vel_dof] = AMPL * np.sin(2 * np.pi * FREQ * data.time)
        mujoco.mj_step(model, data)
    viewer.launch(model=model, data=data)

print("Done.")
