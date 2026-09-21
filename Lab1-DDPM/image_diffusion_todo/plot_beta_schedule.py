"""
Plot beta_t and alpha_bar_t (= prod of alpha_1..t) against timestep t for the
three beta schedules (linear, quad, cosine) implemented in scheduler.py.

This directly reuses BaseScheduler so the plotted curves always match
whatever add_noise() actually uses for the forward process -- useful both as
a sanity check on the cosine-schedule TODO and as a report figure showing how
much noise each schedule injects at a given t.

Usage:
    python plot_beta_schedule.py [--num_train_timesteps 1000] [--beta_1 1e-4] [--beta_T 0.02] [--out beta_schedule.png]
"""
import argparse

import matplotlib.pyplot as plt

from scheduler import BaseScheduler

MODES = ["linear", "quad", "cosine"]


def main(args):
    fig, (ax_beta, ax_abar) = plt.subplots(1, 2, figsize=(12, 4.5))

    for mode in MODES:
        sched = BaseScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            mode=mode,
        )
        t = range(args.num_train_timesteps)
        betas = sched.betas.numpy()
        alphas_cumprod = sched.alphas_cumprod.numpy()

        ax_beta.plot(t, betas, label=mode)
        ax_abar.plot(t, alphas_cumprod, label=mode)

    ax_beta.set_title(r"$\beta_t$ vs. timestep $t$")
    ax_beta.set_xlabel("t")
    ax_beta.set_ylabel(r"$\beta_t$")
    ax_beta.legend()
    ax_beta.grid(alpha=0.3)

    ax_abar.set_title(r"$\bar{\alpha}_t$ vs. timestep $t$")
    ax_abar.set_xlabel("t")
    ax_abar.set_ylabel(r"$\bar{\alpha}_t$")
    ax_abar.legend()
    ax_abar.grid(alpha=0.3)

    if args.xlim_max is not None:
        # Display-only zoom: betas/alphas_cumprod are still computed over the
        # full [0, num_train_timesteps) range above (so add_noise's actual
        # schedule is unaffected) -- this just crops the singularity at
        # t -> T out of the plotted view for readability.
        ax_beta.set_xlim(0, args.xlim_max)
        ax_abar.set_xlim(0, args.xlim_max)

    if args.beta_ylim_max is not None:
        # Also display-only: caps the beta_t y-axis so the cosine
        # near-t=T spike (up to 0.999) doesn't squash the rest of the curves.
        ax_beta.set_ylim(0, args.beta_ylim_max)

    fig.suptitle(
        f"Noise schedules (T={args.num_train_timesteps}, "
        f"beta_1={args.beta_1}, beta_T={args.beta_T})"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="beta_schedule.png")
    parser.add_argument(
        "--xlim_max",
        type=int,
        default=None,
        help="If set, crop the plotted x-axis to [0, xlim_max] (display only "
        "-- the schedule itself is still computed over the full range). "
        "Useful to zoom past the cosine schedule's beta->0.999 singularity "
        "near t=T.",
    )
    parser.add_argument(
        "--beta_ylim_max",
        type=float,
        default=None,
        help="If set, cap the beta_t subplot's y-axis to [0, beta_ylim_max] "
        "(display only). E.g. 0.1 to keep the cosine schedule's near-t=T "
        "spike (up to 0.999) from squashing the rest of the curves.",
    )
    args = parser.parse_args()
    main(args)
