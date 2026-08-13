# Local headless-render smoke

Both bounded local failure-video probes correctly failed before any rollout
cell was written:

- `MUJOCO_GL=glx`: no X11 `DISPLAY`; MuJoCo raised `gladLoadGL error`.
- `MUJOCO_GL=egl`: the development host did not expose an EGL device with the
  required `PLATFORM_DEVICE` extension.

The empty `*_STARTED.json` markers are retained as evidence of those attempts.
Qualification requests exactly one PAI rendering GPU and saves every failed
qualification video there; no second idle GPU is reserved.
