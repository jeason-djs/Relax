# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import atexit
import json
import os
import signal
import sys
from pathlib import Path

import ray
import yaml
from ray import serve

from relax.utils import try_import_telemetry_hook


# Optional telemetry hook: import before Controller so it can install patches.
# Missing or broken hooks must not change training behavior.
try_import_telemetry_hook()

from relax.core.controller import Controller  # noqa: E402
from relax.utils.arguments import parse_args  # noqa: E402
from relax.utils.logging_utils import get_logger  # noqa: E402
from relax.utils.tracking_utils import init_tracking  # noqa: E402
from relax.utils.utils import post_process_env  # noqa: E402


cur_file_dir = Path(__file__).absolute().parent.parent.parent
logger = get_logger(__name__)

# Global reference so signal handlers / atexit can reach the controller.
_ctrl: Controller | None = None
_shutdown_done = False


def _hard_exit(code: int):
    """Exit without running Python/native extension destructors."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


def _graceful_shutdown(sig=None, frame=None, exit_code: int | None = None):
    """Shut down SGLang engines and Ray on SIGTERM / SIGINT / atexit."""
    global _shutdown_done

    if sig is not None:
        exit_code = 128 + sig

    if _shutdown_done:
        if exit_code is not None:
            _hard_exit(exit_code)
        return

    _shutdown_done = True

    sig_name = signal.Signals(sig).name if sig else "atexit"
    logger.info(f"Graceful shutdown triggered ({sig_name}) — cleaning up SGLang engines...")

    if _ctrl is not None:
        try:
            _ctrl.shutdown()
        except Exception as e:
            logger.warning(f"Controller shutdown error during {sig_name}: {e}")

    if ray.is_initialized():
        try:
            serve.shutdown()
            ray.shutdown()
            logger.info("Ray shutdown successfully")
        except Exception as e:
            logger.warning(f"Ray shutdown error during {sig_name}: {e}")

    if exit_code is not None:
        _hard_exit(exit_code)


def main(args):
    global _ctrl

    # Load runtime_env from config so we can both pass it to ray.init and
    # explicitly to the Serve deployment. Ensure it's available even if Ray
    # is already initialized.
    with open(os.path.join(cur_file_dir, "configs/env.yaml")) as file:
        runtime_env = yaml.safe_load(file)

    runtime_env = post_process_env(args, runtime_env)
    attestation_dir = os.environ.get("TASK22_RUNTIME_ATTESTATION_DIR")
    if attestation_dir:
        submitted_runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
        submitted_env_vars = submitted_runtime_env.get("env_vars")
        if not isinstance(submitted_env_vars, dict):
            raise RuntimeError("Task 22 runtime env must contain an env_vars object")
        runtime_env["env_vars"].update(submitted_env_vars)
        runtime_env["env_vars"]["RUNTIME_ENV_JSON"] = os.environ["RUNTIME_ENV_JSON"]
        runtime_env["env_vars"]["TASK22_RUNTIME_ATTESTATION_DIR"] = attestation_dir
        if working_dir := submitted_runtime_env.get("working_dir"):
            runtime_env["working_dir"] = working_dir
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.environ.get("TASK22_RAY_JOB_RUNTIME_ENV_APPLIED") == "1":
            ray.init()
        else:
            ray.init(runtime_env=runtime_env)
        logger.info("Ray initialized successfully")
        try:
            serve.start(
                http_options={"host": "0.0.0.0", "port": "8000"},
                detached=True,
            )
        except RuntimeError:
            pass

    if attestation_dir:
        from relax.utils.task22_runtime_attestation import write_attestation

        runtime_env_json = os.environ["RUNTIME_ENV_JSON"]
        input_manifest_path = os.environ["TASK22_INPUT_MANIFEST"]
        input_roots_json = os.environ["TASK22_INPUT_ROOTS_JSON"]
        write_attestation(
            attestation_dir,
            "driver",
            runtime_env_json,
            input_manifest_path,
            input_roots_json,
        )

    # init_tracking must run after serve.start() (metrics adapter probes Ray
    # Serve for the /metrics endpoint) and before Controller() (wandb primary
    # writes wandb_run_id into args, which then propagates to remote actors).
    init_tracking(args)

    ctrl = Controller(args, runtime_env)
    _ctrl = ctrl

    # Register signal handlers so that `ray job stop` (SIGTERM) triggers cleanup.
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(_graceful_shutdown)

    try:
        ctrl.training_loop()
    except Exception as e:
        logger.exception(f"Training loop failed with error: {e}")
        _graceful_shutdown(exit_code=1)

    logger.info("Main func successfully")
    # Gracefully shut down SGLang engine processes before tearing down Ray Serve.
    _graceful_shutdown(exit_code=0)


if __name__ == "__main__":
    args = parse_args()
    main(args)
