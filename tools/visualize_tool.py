"""Offline drawing of configured tool capsules; no device connections."""

import argparse
from pathlib import Path

from visualize_capsule_pose import ROOT, surface, plt, np, HardwareConfig
from matplotlib.patches import Polygon, Patch
from scipy.spatial import ConvexHull


def render(config, width_mm, output):
    if not np.isfinite(width_mm) or width_mm < 0:
        raise ValueError("width must be finite and nonnegative")
    model = config.build_kinematics()
    links = [link for link in model.links if link.frame == "tool"]
    if not links:
        raise ValueError("no tool capsules configured")
    colors = {"body": "#718096", "first": "#189b91", "second": "#d38c26"}
    capsules = []
    for link in links:
        shift = link.opening_axis * width_mm / 1000
        color = colors["body" if link.contact_group != "finger" else
                       "first" if "first" in link.name else "second"]
        capsules.append((link.start + shift, link.end + shift, link.radius, color))
    grasp = model.tool_from_grasp[:3, 3] * 1000
    fig = plt.figure(figsize=(15, 8), facecolor="white")
    ax = fig.add_subplot(131, projection="3d")
    for a, b, r, color in capsules:
        xyz = surface(a, b, r)
        ax.plot_surface(*np.moveaxis(xyz, -1, 0), color=color, alpha=.65,
                        linewidth=0, shade=True)
    ax.scatter(0, 0, 0, color="black", marker="+", s=100)
    ax.scatter(*grasp, color="#c42d42", marker="x", s=80)
    for i, color in enumerate(("#c42d42", "#258c42", "#3269bd")):
        direction = np.eye(3)[i] * 45
        ax.quiver(0, 0, 0, *direction, color=color)
    ax.set(xlim=(-90, 90), ylim=(-90, 90), zlim=(-40, 270),
           xlabel="Tool X (mm)", ylabel="Tool Y (mm)", zlabel="Tool Z (mm)",
           title="Configured capsules in 3D")
    ax.set_box_aspect((180, 180, 310))
    ax.view_init(elev=15, azim=-55)
    angles = np.linspace(0, 2*np.pi, 96)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    for panel, axis, title in ((2, 1, "Front: opening along tool Y"),
                               (3, 0, "Side: fingers overlap in projection")):
        ax = fig.add_subplot(1, 3, panel)
        for a, b, r, color in capsules:
            points = np.vstack((a[[axis, 2]] + r*circle,
                                b[[axis, 2]] + r*circle)) * 1000
            ax.add_patch(Polygon(points[ConvexHull(points).vertices],
                                 facecolor=color, edgecolor=color, alpha=.45))
            ax.plot([a[axis]*1000, b[axis]*1000], [a[2]*1000, b[2]*1000],
                    color=color, linewidth=1)
        ax.scatter(0, 0, color="black", marker="+", s=100, zorder=5)
        ax.scatter(grasp[axis], grasp[2], color="#c42d42", marker="x", s=80, zorder=5)
        ax.axhline(0, color="black", linewidth=.7, linestyle=":")
        ax.axhline(grasp[2], color="#c42d42", linewidth=.7, linestyle=":")
        ax.set(xlim=(-95, 95), ylim=(-40, 270), aspect="equal", title=title,
               xlabel=f"Tool {'Y' if axis == 1 else 'X'} (mm)", ylabel="Tool Z (mm)")
        ax.grid(alpha=.2)
    tip = max(max(a[2], b[2]) + r for a, b, r, _ in capsules)*1000
    fig.suptitle(f"Simplified tool collision model | assumed opening {width_mm:g} mm", fontsize=17)
    fig.legend([Patch(color=c) for c in colors.values()],
               ["Body", "First segments", "Second segments / pads"],
               loc="lower center", bbox_to_anchor=(.5, .10), ncol=3)
    fig.text(.5, .055, f"Black +: flange origin. Red x: configured grasp point ({grasp[2]:g} mm along Z). "
             f"Capsule envelope tip: {tip:g} mm.", ha="center", fontsize=10)
    fig.text(.5, .025, "Straight sliding fingers assumed; actual linkage rotation is NOT modeled. "
             "Collision geometry preview, not hardware clearance approval.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.025, right=.98, bottom=.19, top=.87, wspace=.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-mm", type=float, default=64.)
    parser.add_argument("--output", type=Path, default=ROOT / "simulation_plots/tool_capsules.png")
    args = parser.parse_args()
    render(HardwareConfig.load(), args.width_mm, args.output)
    print(args.output)
