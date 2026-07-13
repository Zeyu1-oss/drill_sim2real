"""把 point_cloud 从大块 (100帧) 重切成小块 (8帧),消除随机窗口读取的解压放大。
全程流式(每次只持有 ~90MB),低内存;校验通过才算成功。不在本脚本里删除源。

用法: python tools/rechunk_zarr.py SRC.zarr DST.zarr [--pc_chunk 8]
"""
import argparse, os, time
import numpy as np
import zarr
import numcodecs
from numcodecs import Blosc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--pc_chunk", type=int, default=8, help="point_cloud 帧轴块大小")
    ap.add_argument("--block", type=int, default=2048, help="流式拷贝每次读多少帧")
    ap.add_argument("--nthreads", type=int, default=16, help="blosc 压缩线程")
    args = ap.parse_args()

    numcodecs.blosc.set_nthreads(args.nthreads)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    src = zarr.open(args.src, "r")
    assert not os.path.exists(args.dst), f"目标已存在: {args.dst}"
    dst = zarr.open(args.dst, "w")
    d_data = dst.create_group("data")
    d_meta = dst.create_group("meta")

    # meta(小)
    ee = src["meta/episode_ends"][:]
    d_meta.array("episode_ends", ee, chunks=ee.shape, dtype=ee.dtype)
    T = int(src["data/point_cloud"].shape[0])
    print(f"[rechunk] T={T} episodes={len(ee)} pc_chunk={args.pc_chunk}", flush=True)

    # state / action(小)→ 顺手切成 (2048,13)
    for k in ("state", "action"):
        a = src[f"data/{k}"]
        out = d_data.zeros(k, shape=a.shape, chunks=(2048, a.shape[1]),
                           dtype=a.dtype, compressor=comp)
        for s in range(0, T, 65536):
            out[s:s + 65536] = a[s:s + 65536]
        print(f"[rechunk] {k} done", flush=True)

    # point_cloud(大)→ (pc_chunk, P, 3),流式
    pc = src["data/point_cloud"]
    P = pc.shape[1]
    out = d_data.zeros("point_cloud", shape=pc.shape, chunks=(args.pc_chunk, P, 3),
                       dtype=pc.dtype, compressor=comp)
    t0 = time.time()
    for s in range(0, T, args.block):
        e = min(s + args.block, T)
        out[s:e] = pc[s:e]
        if (s // args.block) % 20 == 0:
            done = e / T
            el = time.time() - t0
            eta = el / max(done, 1e-9) * (1 - done)
            print(f"[rechunk] pc {e}/{T} ({done*100:4.1f}%) "
                  f"el={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
    print(f"[rechunk] point_cloud done in {(time.time()-t0)/60:.1f}m", flush=True)

    # ===== 校验:无损,逐点必须完全相等 =====
    print("[verify] checking shapes / episode_ends / random frames ...", flush=True)
    assert out.shape == pc.shape, (out.shape, pc.shape)
    assert np.array_equal(d_meta["episode_ends"][:], ee)
    rng = np.random.RandomState(0)
    idx = np.unique(rng.randint(0, T, size=64))
    ok = True
    for i in idx:
        if not np.array_equal(out[i], pc[i]):
            print(f"[verify] MISMATCH at frame {i}", flush=True)
            ok = False
            break
    for k in ("state", "action"):
        if not np.array_equal(d_data[k][idx], src[f"data/{k}"][idx]):
            print(f"[verify] MISMATCH in {k}", flush=True)
            ok = False
    if ok:
        print("[verify] OK — 所有抽查帧逐点相等,重切块成功", flush=True)
        print("RECHUNK_SUCCESS", flush=True)
    else:
        print("RECHUNK_FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
