import mujoco
import numpy as np
import time

# Load the XML model
with open('pendulum.xml') as f:
    xml = f.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

SIM_DURATION = 10.0  # seconds
FREQ_HZ = 0.5        # frequency of commanded velocity waveform
VEL_AMPL = 2     # amplitude of commanded velocity

def simulate_with_viewer():
    import mujoco.viewer as viewer
    # Prefer passive viewer so we control stepping

    with viewer.launch_passive(model, data) as v:
        start = data.time
        while v.is_running() and (data.time - start) < SIM_DURATION:
            if model.nv > 0:
                data.qvel[0] = VEL_AMPL * np.sin(2*np.pi*FREQ_HZ * data.time)
            mujoco.mj_step(model, data)
            v.sync()  # push updated state to viewer
            time.sleep(0.05)


def simulate_headless():
    print("Viewer unavailable, running headless.")
    start = data.time
    while (data.time - start) < SIM_DURATION:
        if model.nv > 0:
            data.qvel[0] = VEL_AMPL * np.sin(2*np.pi*FREQ_HZ * data.time)
        mujoco.mj_step(model, data)
        if int(data.time / model.opt.timestep) % 200 == 0:
            print(f"t={data.time:.3f} qpos0={data.qpos[0]:.3f} qvel0={data.qvel[0]:.3f}")

if __name__ == '__main__':
    try:
        simulate_with_viewer()
    except Exception as e:
        print("Viewer error:", e)
        simulate_headless()

