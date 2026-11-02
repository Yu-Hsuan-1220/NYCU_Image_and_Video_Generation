"""
Extra measurements and figures for the Lab4 report.

- Task 1: retrain the 2D FM with the notebook config and plot sampling trajectories.
- Task 3: trajectory straightness of base FM vs 1-RF, and endpoint predictions along the ODE.
- Task 4: wall-clock timing of every pipeline stage in one process.
- Sample grids for Tasks 2-4 from the images produced by the run scripts.

Usage: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/report_analysis.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching
from image_common.instaflow import InstaFlowModel

BASE_RUN = ROOT / "results/cfg_fm-11-01-023216"
RF_RUN = ROOT / "results/rectified_fm_1-11-01-151854"
INSTA_RUN = ROOT / "results/instaflow-11-01-233319"
OUT = ROOT / "report/Figure"
DEVICE = "cuda:0"
OUT.mkdir(parents=True, exist_ok=True)


def load_fm(run):
    fm = FlowMatching(None, None)
    fm.load(run / "last.ckpt")
    return fm.eval().to(DEVICE)


# ---------------------------------------------------------------- Task 1
def task1():
    sys.path.insert(0, str(ROOT / "task1_2d_flow_matching"))
    from chamferdist import chamfer_distance
    from dataset import TwoDimDataClass, get_data_iterator
    from fm import FlowMatching as FM2D, FMScheduler as Sched2D
    from network import SimpleNet

    torch.manual_seed(0)
    np.random.seed(0)
    target_ds = TwoDimDataClass("swiss_roll", N=1000000, batch_size=128)
    fm = FM2D(SimpleNet(2, 2, [128, 128, 128], 1000), Sched2D(1000, 0.01)).to(DEVICE)
    opt = torch.optim.Adam(fm.network.parameters(), lr=5e-4)
    it = get_data_iterator(torch.utils.data.DataLoader(target_ds, batch_size=128))
    for _ in range(5000):
        loss = fm.get_loss(next(it).to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()

    traj = fm.sample((2048, 2), num_inference_timesteps=50, return_traj=True)
    traj = torch.stack([x.cpu() for x in traj]).numpy()  # [T+1, N, 2]
    cd = chamfer_distance(traj[-1], target_ds[:2048].numpy())

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for i in range(300):
        axs[0].plot(traj[:, i, 0], traj[:, i, 1], color="gray", alpha=0.25, lw=0.6)
    axs[0].scatter(traj[0, :300, 0], traj[0, :300, 1], s=4, c="tab:blue", label="$x_0$ (prior)")
    axs[0].scatter(traj[-1, :300, 0], traj[-1, :300, 1], s=4, c="tab:red", label="$x_1$ (generated)")
    axs[0].set_title("Sampling trajectories (50 Euler steps)")
    axs[0].legend(loc="lower right", fontsize=8)
    ref = target_ds[:2048].numpy()
    axs[1].scatter(ref[:, 0], ref[:, 1], s=3, alpha=0.5, label="target")
    axs[1].scatter(traj[-1, :, 0], traj[-1, :, 1], s=3, alpha=0.5, label="generated")
    axs[1].set_title(f"Generated vs. target (CD = {cd:.2f})")
    axs[1].legend(loc="lower right", fontsize=8)
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    plt.tight_layout()
    plt.savefig(OUT / "task1_traj.png", dpi=200)
    plt.close()
    return {"rerun_chamfer_distance": float(cd)}


# ---------------------------------------------------------------- Task 3
@torch.no_grad()
def straightness(fm, x0, labels, steps=20, cfg=7.5):
    """Liu et al. straightness: E_t ||(x_1 - x_0) - v_t||^2, averaged per pixel."""
    traj = fm.sample(x0.shape, num_inference_timesteps=steps, class_label=labels,
                     guidance_scale=cfg, return_traj=True, x0=x0)
    traj = torch.stack([x.to(DEVICE) for x in traj])  # [T+1, B, C, H, W]
    v = (traj[1:] - traj[:-1]) * steps
    disp = traj[-1] - traj[0]
    s = ((disp[None] - v) ** 2).mean().item()
    t = torch.arange(steps, device=DEVICE).view(-1, 1, 1, 1, 1) / steps
    x1_hat = traj[:-1] + (1 - t) * v  # endpoint the model is heading to at each step
    return s, x1_hat


def task3():
    base, rf = load_fm(BASE_RUN), load_fm(RF_RUN)
    torch.manual_seed(1)
    x0 = torch.randn(256, 3, 64, 64, device=DEVICE)
    labels = torch.randint(1, 4, (256,), device=DEVICE)
    # Straightness is a property of the learned (unguided) field; CFG 7.5 is
    # reported too because it amplifies the cond/uncond gap at t=0.
    results = {}
    for cfg in [1.0, 7.5]:
        s_base, xh_base = straightness(base, x0, labels, cfg=cfg)
        s_rf, xh_rf = straightness(rf, x0, labels, cfg=cfg)
        results[f"straightness_cfg{cfg}"] = {"base_fm": s_base, "1rf": s_rf}
    s_base, xh_base = straightness(base, x0, labels, cfg=1.0)
    s_rf, xh_rf = straightness(rf, x0, labels, cfg=1.0)

    cols = [0, 2, 5, 10, 15, 19]
    fig, axs = plt.subplots(4, len(cols), figsize=(len(cols) * 1.2, 4 * 1.25))
    for r, (name, xh, idx) in enumerate([("Base FM", xh_base, 0), ("1-RF", xh_rf, 0),
                                         ("Base FM", xh_base, 3), ("1-RF", xh_rf, 3)]):
        for c, k in enumerate(cols):
            axs[r, c].imshow(tensor_to_pil_image(xh[k, idx].cpu(), single_image=True))
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(f"t={k / 20:.2f}", fontsize=8)
            if c == 0:
                axs[r, c].set_ylabel(name, fontsize=8)
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "task3_x1hat.png", dpi=200)
    plt.close()
    return results


# ---------------------------------------------------------------- Task 4
@torch.no_grad()
def time_it(fn, n=500, bs=128):
    fn(8)  # warm-up
    torch.cuda.synchronize()
    start = time.time()
    for s in range(0, n, bs):
        fn(min(bs, n - s))
    torch.cuda.synchronize()
    return time.time() - start


def task4():
    base, rf = load_fm(BASE_RUN), load_fm(RF_RUN)
    insta = InstaFlowModel(None)
    insta.load(INSTA_RUN / "last.ckpt")
    insta = insta.eval().to(DEVICE)
    lab = lambda b: torch.randint(1, 4, (b,), device=DEVICE)
    shape = lambda b: (b, 3, 64, 64)
    stages = {
        "Base FM (50 steps, CFG 7.5)": lambda b: base.sample(shape(b), 50, class_label=lab(b), guidance_scale=7.5),
        "1-RF (20 steps, CFG 1.5)": lambda b: rf.sample(shape(b), 20, class_label=lab(b), guidance_scale=1.5),
        "InstaFlow (1 step)": lambda b: insta.sample(shape(b), class_label=lab(b)),
    }
    return {k: time_it(f) for k, f in stages.items()}


ABLATIONS = {  # scripts/run_task4_ablation.sh
    "L2 + LPIPS, 1-RF init": INSTA_RUN,
    "L2 only, 1-RF init": ROOT / "results/instaflow-11-02-145618",
    "L2 + LPIPS, random init": ROOT / "results/instaflow-11-02-194117",
}


@torch.no_grad()
def task4_ablation(n=8):
    """One-step samples of every ablation from the same noise and labels, and
    the training-progress samples (saved by train_instaflow.py) with and without init."""
    torch.manual_seed(2)
    x0 = torch.randn(n, 3, 64, 64, device=DEVICE)
    labels = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3], device=DEVICE)[:n]
    fig, axs = plt.subplots(len(ABLATIONS), 1, figsize=(n * 0.9, len(ABLATIONS) * 1.0))
    for ax, (name, run) in zip(axs, ABLATIONS.items()):
        insta = InstaFlowModel(None)
        insta.load(run / "last.ckpt")
        insta = insta.eval().to(DEVICE)
        imgs = tensor_to_pil_image(insta.network(x0, torch.zeros(n, device=DEVICE), class_label=labels) + x0)
        ax.imshow(np.concatenate([np.asarray(im) for im in imgs], axis=1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name.replace(", ", "\n"), fontsize=7, rotation=0, ha="right", va="center")
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "task4_ablation_samples.png", dpi=200, bbox_inches="tight")
    plt.close()

    steps = [0, 1000, 5000, 20000, 99800]
    runs = {"1-RF init": INSTA_RUN, "Random init": ABLATIONS["L2 + LPIPS, random init"]}
    fig, axs = plt.subplots(len(runs), len(steps), figsize=(len(steps) * 1.4, len(runs) * 1.5))
    for r, (name, run) in enumerate(runs.items()):
        for c, s in enumerate(steps):
            im = [np.asarray(Image.open(run / f"step={s}-{i}.png")) for i in range(4)]
            axs[r, c].imshow(np.concatenate([np.concatenate(im[:2], 1), np.concatenate(im[2:], 1)], 0))
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(f"step {s / 1000:g}k" if s else "step 0", fontsize=8)
            if c == 0:
                axs[r, c].set_ylabel(name, fontsize=8)
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "task4_ablation_progress.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- grids
def grid(rows, path, n=8):
    """rows: list of (label, image dir, filename pattern)."""
    fig, axs = plt.subplots(len(rows), 1, figsize=(n * 0.9, len(rows) * 1.0))
    for ax, (label, d, pat) in zip(np.atleast_1d(axs), rows):
        imgs = [np.asarray(Image.open(d / pat.format(i))) for i in range(n)]
        ax.imshow(np.concatenate(imgs, axis=1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def grids():
    be, re, ie = BASE_RUN / "eval", RF_RUN / "eval", INSTA_RUN / "eval"
    grid([(f"CFG {c}", be / f"cfg{c}_steps20", "{}.png") for c in ["1.0", "3.0", "7.5"]],
         OUT / "task2_cfg.png")
    grid([(f"{s} steps", be / f"cfg7.5_steps{s}", "{}.png") for s in [10, 20, 50]],
         OUT / "task2_steps.png")
    grid([("Base FM\n5 steps", re / "base_fm_steps5", "sample_{:04d}.png"),
          ("1-RF\n5 steps", re / "rectified_1_steps5", "sample_{:04d}.png"),
          ("Base FM\n20 steps", re / "base_fm_steps20", "sample_{:04d}.png"),
          ("1-RF\n20 steps", re / "rectified_1_steps20", "sample_{:04d}.png")],
         OUT / "task3_samples.png")
    grid([("Base FM\n50 steps", be / "cfg7.5_steps50", "{}.png"),
          ("1-RF\n20 steps", ie / "2rf_20steps", "sample_{:04d}.png"),
          ("InstaFlow\n1 step", ie / "instaflow_1step", "sample_{:04d}.png")],
         OUT / "task4_samples.png")
    # Few-step sampling with and without CFG (eval_cfg* dirs from evaluate_rectified.py --cfg_scale)
    grid([("Base FM\nCFG 7.5, 1 step", RF_RUN / "eval_cfg7.5_few/base_fm_steps1", "sample_{:04d}.png"),
          ("1-RF\nCFG 7.5, 1 step", RF_RUN / "eval_cfg7.5_few/rectified_1_steps1", "sample_{:04d}.png"),
          ("1-RF\nCFG 7.5, 2 steps", RF_RUN / "eval_cfg7.5_few/rectified_1_steps2", "sample_{:04d}.png"),
          ("1-RF\nCFG 1.0, 1 step", RF_RUN / "eval_cfg1.0/rectified_1_steps1", "sample_{:04d}.png"),
          ("1-RF\nCFG 1.0, 2 steps", RF_RUN / "eval_cfg1.0/rectified_1_steps2", "sample_{:04d}.png")],
         OUT / "task3_few_steps.png")
    for name, run in [("task2", BASE_RUN), ("task3", RF_RUN), ("task4", INSTA_RUN)]:
        Image.open(run / "loss.png").save(OUT / f"{name}_loss.png")


if __name__ == "__main__":
    results = {}
    results.update(task1())
    results.update(task3())
    results["timing_500_samples_sec"] = task4()
    task4_ablation()
    grids()
    print(json.dumps(results, indent=2))
    with open(OUT / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
