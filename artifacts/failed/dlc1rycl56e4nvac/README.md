# Failed first PAI attempt

PAI job `dlc1rycl56e4nvac` failed after 85 seconds, before training. A clean
worker imported official LIBERO without a pre-existing user config; LIBERO
prompted for an interactive dataset directory and batch stdin returned EOF.

The traceback is in `run.log`. The experiment was fixed by checking in a
deterministic LIBERO config and setting `LIBERO_CONFIG_PATH`. The replacement
job `dlc6sr1fu466f1g9` succeeded. This directory is retained for transparent
failure provenance.
