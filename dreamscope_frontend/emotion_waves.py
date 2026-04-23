import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Wave generation
# ---------------------------------------------------------------------------

def _make_wave(
    x: np.ndarray,
    score: float,
    freq: float,
    phase: float,
    max_amp: float,
) -> np.ndarray:
    """
    Returns the y-values of the wave with a sinusoidal envelope.
    """
    envelope = np.sin(x * np.pi)
    t = x * 2 * np.pi * freq + phase
    return np.sin(t) * score * max_amp * envelope


# ---------------------------------------------------------------------------
# Alpha gradient fill
# ---------------------------------------------------------------------------

def _fill_wave_gradient(
    ax,
    x: np.ndarray,
    y_wave: np.ndarray,
    mid_y: float,
    color_rgb: tuple,
    n_strips: int = 80,
):
    """
    Fill the area between mid_y and y_wave with an opacity gradient.
    The alpha value is maximum at the peaks (high |y - mid_y|) and zero at mid_y.
    Divide the total amplitude into n_strips horizontal bands.
    For each band [y_lo, y_hi], use fill_between between y_lo and y_hi, clamped by y_wave, with an alpha proportional to the band height.
    Process positive and negative parts separately.
    """
    amp_max = np.max(np.abs(y_wave - mid_y))
    if amp_max < 1e-6:
        return

    r, g, b = color_rgb

    for sign in (+1, -1):
        # Bands from mid_y to +amp_max (sign=+1) or -amp_max (sign=-1)
        levels = np.linspace(0, sign * amp_max, n_strips + 1)

        for i in range(n_strips):
            edge_a = mid_y + levels[i]
            edge_b = mid_y + levels[i + 1]

            # We always want y_lo < y_hi for fill_between
            y_lo = min(edge_a, edge_b)
            y_hi = max(edge_a, edge_b)

            # Alpha proportional to the distance from the band center
            band_center_dist = (abs(levels[i]) + abs(levels[i + 1])) / 2
            alpha = (band_center_dist / amp_max) * 0.85

            if sign == +1:
                # Positive part: fill where y_wave exceeds y_lo
                mask = y_wave >= y_lo
                if not np.any(mask):
                    continue
                # y2 = min(y_wave, y_hi) to stay within the band
                y2 = np.minimum(y_wave, y_hi)
                ax.fill_between(
                    x, y_lo, y2,
                    where=mask,
                    color=(r, g, b, alpha),
                    linewidth=0,
                    interpolate=True,
                )
            else:
                # Negative part: fill where y_wave is below y_hi
                mask = y_wave <= y_hi
                if not np.any(mask):
                    continue
                # y2 = max(y_wave, y_lo) to stay within the band
                y2 = np.maximum(y_wave, y_lo)
                ax.fill_between(
                    x, y2, y_hi,
                    where=mask,
                    color=(r, g, b, alpha),
                    linewidth=0,
                    interpolate=True,
                )


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def plot_emotion_waves(
    emotions: list,
    figsize: tuple = (9, 5),
    is_dark_mode: bool = True,
    x_steps: int = 1000,
) -> plt.Figure:
    """
    Generate the Matplotlib plot of emotional waves.

    Parameters
    ----------
    emotions: list of dict
        Each dict must contain:
          - "label"   : str         — emotion name
          - "score"   : float       — score from 0.0 to 1.0
          - "RGB"     : tuple(int)  — RGB color (0-255), e.g.: (220, 50, 30)
        Optional fields:
          - "freq"    : float       — wave frequency (default: auto based on rank)
          - "phase"   : float       — initial phase (default: auto based on rank)

    figsize   : tuple — figure size (width, height) in inches
    dark_bg   : bool  — dark background (True) or light background (False)
    x_steps   : int   — horizontal resolution of the wave
    """

    default_freqs  = [1.8, 3.2, 5.1, 1.1, 2.5, 4.0, 6.2, 2.0]
    default_phases = [0.0, 0.9, 1.7, 2.4, 0.5, 1.2, 3.1, 0.3]

    for i, e in enumerate(emotions):
        e.setdefault("freq",  default_freqs[i  % len(default_freqs)])
        e.setdefault("phase", default_phases[i % len(default_phases)])

    mid_color = "#FFFFFF" if is_dark_mode else "#000000"
    txt_color = "#EEEEF5" if is_dark_mode else "#333333"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    x = np.linspace(0, 1, x_steps)
    mid_y   = 0.0
    max_amp = 1.0

    for e in emotions:
        # RGB normalization 0-255 → 0-1
        color_rgb = e['RGB'] # valence_to_rgb(e["valence"])
        color_rgb_normalized = tuple(c / 255 for c in color_rgb)
        y_wave = _make_wave(x, e["score"], e["freq"], e["phase"], max_amp)

        # Area with gradient
        _fill_wave_gradient(ax, x, y_wave, mid_y, color_rgb_normalized, n_strips=80)

        # Outline
        ax.plot(
            x, y_wave,
            color=color_rgb_normalized,
            linewidth=1.6,
            alpha=0.92,
        )

    # Central line
    ax.axhline(
        mid_y,
        color=mid_color,
        linewidth=0.5,
        linestyle="--",
        alpha=0.2,
    )

    # Horizontal legend at the top, centered on the figure
    legend_elements = []
    for e in emotions:
        color_rgb = tuple(c / 255.0 for c in e['RGB'])
        patch = plt.Line2D(
            [0], [0],
            color=color_rgb,
            linewidth=4.0,
            solid_capstyle="round",
            label=f"{e['label']}  {int(round(e['score'] * 100))}%",
        )
        legend_elements.append(patch)

    plt.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(emotions),
        framealpha=0.0,
        labelcolor=txt_color,
        fontsize=18,
        handlelength=2.2,
        handleheight=1.0,
        borderpad=0.6,
        columnspacing=2.5,
        handletextpad=1.2,
        prop={'weight': 'bold'}
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-max_amp * 1.00, max_amp * 1.00)
    ax.axis("off")

    # --- Margins: adjusted for the top legend and compact on sides
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(top=0.82)
    return fig


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_emotions = [
        {"label": "Sadness",        "score": 0.59, "RGB": (30,  144, 255)},
        {"label": "Disappointment", "score": 0.11, "RGB": (255, 200,  50)},
        {"label": "Nervousness",    "score": 0.10, "RGB": (200,  50, 100)},
        {"label": "Fear",           "score": 0.08, "RGB": (220,  80,  30)},
    ]

    fig = plot_emotion_waves(sample_emotions, dark_bg=True)
    plt.show()
