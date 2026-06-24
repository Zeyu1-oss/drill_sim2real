import numpy as np
import zarr
import sys


data_path = '/home/zeyu/inspire_drill/data/inspire_drill_dp3_target_test_new111.zarr'
episode_idx = 17
frame_in_ep = 1

z = zarr.open(data_path, mode='r')
episode_ends = z['meta']['episode_ends'][:]
n_ep = len(episode_ends)
starts = np.concatenate([[0], episode_ends[:-1]])      # 每条 episode 的全局起始帧

assert 0 <= episode_idx < n_ep, f"episode 越界:{episode_idx},共 {n_ep} 条(0..{n_ep-1})"
s, e = int(starts[episode_idx]), int(episode_ends[episode_idx])
ep_len = e - s
t = frame_in_ep + ep_len if frame_in_ep < 0 else frame_in_ep   # 支持负索引(-1=最后一帧)
assert 0 <= t < ep_len, f"帧越界:episode {episode_idx} 只有 {ep_len} 帧(0..{ep_len-1})"
frame_idx = s + t      # (episode, 帧内偏移) -> 全局帧索引

pointcloud = z['data']['point_cloud'][frame_idx]
print(f"episode {episode_idx}(全局帧 [{s},{e}), 长度 {ep_len})的第 {t} 帧 -> 全局帧 {frame_idx}")
print(f"pointcloud shape = {pointcloud.shape}")
print(f"Point cloud range: x=[{pointcloud[:,0].min():.3f}, {pointcloud[:,0].max():.3f}], "
      f"y=[{pointcloud[:,1].min():.3f}, {pointcloud[:,1].max():.3f}], "
      f"z=[{pointcloud[:,2].min():.3f}, {pointcloud[:,2].max():.3f}]")

# 分段(按点数自动判断布局):
#   2560 = camera 2048 + ground 512(camera 模式)
#   3810 = camera 2048 + robot 1250 + ground 512(robot 模式)
# 地面恒为最后 512 点;robot 段(若有)在 camera 与 ground 之间。
N = pointcloud.shape[0]
GROUND_N = 512
CAM_N = 2048
segments = [("camera", 0, CAM_N)]
if N == CAM_N + GROUND_N:                      # camera 模式 2560
    segments.append(("ground", CAM_N, N))
elif N > CAM_N + GROUND_N:                     # robot / robot_drill 模式
    segments.append(("robot+drill", CAM_N, N - GROUND_N))
    segments.append(("ground", N - GROUND_N, N))
else:
    segments = [("all", 0, N)]

# 每段一种颜色,便于在 HTML 里一眼区分(地面=红色)
seg_color = {"camera": (0, 200, 255), "robot+drill": (0, 255, 0),
             "ground": (255, 60, 60), "all": (0, 200, 255)}
for name, a, b in segments:
    seg = pointcloud[a:b]
    print(f"  {name:12s}[{a}:{b}] x=[{seg[:,0].min():.3f},{seg[:,0].max():.3f}] "
          f"y=[{seg[:,1].min():.3f},{seg[:,1].max():.3f}] "
          f"z=[{seg[:,2].min():.3f},{seg[:,2].max():.3f}]")

# 每段 RGB(地面红色,一眼可辨)
rgb = np.zeros((N, 3), dtype=np.float32)
for name, a, b in segments:
    rgb[a:b] = seg_color.get(name, (180, 180, 180))

# 直接用 plotly 存 HTML,aspectmode='data' 按真实数据范围自动缩放。
# 不用 Visualizer.save_visualization_to_file —— 它把坐标轴写死成 [-1,1],
# 会把 env-local 坐标里 x>1(到 1.75)的地面点全裁掉。
import plotly.graph_objs as go
import plotly.io as pio
colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in rgb]
trace = go.Scatter3d(
    x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
    mode='markers', marker=dict(size=3, color=colors, opacity=0.9))
fig = go.Figure(data=[trace])
fig.update_layout(scene=dict(aspectmode='data', bgcolor='white'))  # 自动贴合数据,不裁剪
html_path = f'/home/zeyu/inspire_drill/data/viz_pc_ep{episode_idx}_t{t}.html'
pio.write_html(fig, html_path)
print(f"Saved to {html_path}  (camera=青, robot=绿, ground=红;axis 自动缩放,不裁剪)")
