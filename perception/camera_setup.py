
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERCEPTION_HP = None


def load_perception_hp():

    global _PERCEPTION_HP
    if _PERCEPTION_HP is None:
        import importlib.util
        _p = os.path.join(_ROOT, "tasks", "config", "config.py")
        spec = importlib.util.spec_from_file_location("_hp_standalone", _p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _PERCEPTION_HP = m.DEFAULT_HYPERPARAMETERS.perception
    return _PERCEPTION_HP


def ensure_rgb_aov(scene_cfg, cam_names=("cam1", "cam2", "cam3")):

    for name in cam_names:
        c = getattr(scene_cfg, name, None)
        if c is not None and "rgb" not in c.data_types:
            c.data_types = ["rgb"] + list(c.data_types)


def get_cameras(env_unwrapped, chained=False, need_cam3=None):
    # need_cam3: whether cam3 is required. Default None -> follow chained (backward compatible).
    # stage1_only grasp-only collection passes need_cam3=False: cam2 is still the wrist cam but
    # cam3 is not built, and this does not raise.
    if need_cam3 is None:
        need_cam3 = chained
    cam1 = env_unwrapped.scene.sensors.get("cam1")
    cam2 = env_unwrapped.scene.sensors.get("cam2")
    if cam1 is None or cam2 is None:
        raise RuntimeError(
            f"Expected env-managed cameras 'cam1'/'cam2', got "
            f"sensors={list(env_unwrapped.scene.sensors.keys())}")
    cam3 = env_unwrapped.scene.sensors.get("cam3")
    if need_cam3 and cam3 is None:
        raise RuntimeError(
            f"cam3 (plate camera) required but not created, got "
            f"sensors={list(env_unwrapped.scene.sensors.keys())}")
    return cam1, cam2, cam3


def detect_wrist_cam(env_unwrapped, cam):

    link = cam.cfg.prim_path.split("/")[-2]
    on_body = link in list(env_unwrapped.franka.body_names)
    if on_body:
        print(f"[WRIST-CAM] camera mounted on link '{link}', extrinsics from FK "
              f"(cam.data.pos_w is frozen/unusable)", flush=True)
    return on_body


def apply_render_settings(step_dt):

    import carb
    settings = carb.settings.get_settings()
    settings.set("/app/player/useFixedTimeStepping", True)
    settings.set("/app/player/targetFrameRate", int(1.0 / step_dt))
    settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
    settings.set("/app/runLoops/rendering_0/rateLimit", 60)
    settings.set("/physics/updateToUsd", False)
    settings.set("/physics/updateParticlesToUsd", False)
    settings.set("/app/runLoops/main/enabled", True)
    settings.set("/app/runLoops/main/syncTooFast", True)
    return settings


def update_cameras(dt, *cams):
    for c in cams:
        if c is not None:
            c.update(dt)


def setup_ground(num_envs, workspace, device, verbose=True, num_points=None):
    from perception.robot_pointcloud import make_ground_points
    perc = load_perception_hp()
    # num_points: None/negative -> use config default; 0 -> disable (no ground); positive -> that many
    M = perc.ground_points if (num_points is None or int(num_points) < 0) else int(num_points)
    pts = make_ground_points(tuple(workspace), M, perc.ground_z, device)
    batch = pts.unsqueeze(0).expand(num_envs, -1, -1)
    xy_std = float(getattr(perc, "ground_xy_noise_std", 0.0))
    if verbose:
        print(f"[INFO] synthetic ground: {M} points @ z={perc.ground_z} (env-local), "
              f"per-frame xy gaussian jitter sigma={xy_std}m (clamped to region), merged into point_cloud",
              flush=True)
    return batch, M, xy_std
