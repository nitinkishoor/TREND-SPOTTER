"""
Realistic Stock Chart Screenshot Generator for Testing
Generates realistic trading charts (TradingView dark theme style)
with embedded patterns:
  1. Double Top ("Double Header")
  2. Head and Shoulders
  3. Double Bottom ("W" pattern)
  4. Symmetrical Triangle
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_chart_canvas(width_px=1000, height_px=600, dark_theme=True):
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi
    )
    bg_color = "#131722" if dark_theme else "#ffffff"
    grid_color = "#2a2e39" if dark_theme else "#e0e3eb"
    text_color = "#d1d4dc" if dark_theme else "#131722"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Grid
    ax.grid(True, color=grid_color, linestyle="-", linewidth=0.6, alpha=0.7)

    # Styling axes
    ax.tick_params(colors=text_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    return fig, ax, text_color


def draw_candlesticks(ax, opens, highs, lows, closes):
    n = len(opens)
    width = 0.6
    bull_color = "#26a69a"  # TradingView green
    bear_color = "#ef5350"  # TradingView red

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = bull_color if c >= o else bear_color

        # Draw Wick
        ax.plot([i, i], [l, h], color=color, linewidth=1.2, zorder=2)

        # Draw Candle Body
        lower = min(o, c)
        height = max(0.1, abs(c - o))
        rect = patches.Rectangle(
            (i - width / 2, lower),
            width,
            height,
            linewidth=0.8,
            edgecolor=color,
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)


def generate_double_top_chart(output_path="samples/sample_double_top.png"):
    """Generates a realistic chart featuring a Double Top (Double Header) pattern."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    # Base price sequence with two peaks at ~150 and valley at ~132
    # Uptrend -> Peak 1 (150) -> Trough (132) -> Peak 2 (150) -> Breakdown
    segment1 = np.linspace(115, 149, 20)  # Uptrend
    segment2 = np.linspace(149, 132, 12)  # Decline to trough
    segment3 = np.linspace(132, 149.5, 12)  # Rally to second peak
    segment4 = np.linspace(149.5, 122, 18)  # Breakdown

    trend = np.concatenate([segment1, segment2, segment3, segment4])
    noise = np.random.normal(0, 0.8, len(trend))
    closes = trend + noise

    # Generate realistic OHLC from closes
    opens = np.roll(closes, 1)
    opens[0] = closes[0] - 0.5
    highs = np.maximum(opens, closes) + np.random.uniform(0.3, 1.5, len(closes))
    lows = np.minimum(opens, closes) - np.random.uniform(0.3, 1.5, len(closes))

    fig, ax, text_color = create_chart_canvas(1000, 600, dark_theme=True)
    draw_candlesticks(ax, opens, highs, lows, closes)

    ax.set_title(
        "BTC/USDT - 4H | Clean Double Top Setup",
        color=text_color,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(-1, len(closes) + 2)
    ax.set_ylim(min(lows) - 5, max(highs) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated Double Top sample at: {output_path}")


def generate_head_and_shoulders_chart(
    output_path="samples/sample_head_and_shoulders.png",
):
    """Generates a realistic chart with a clear Head and Shoulders pattern."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(101)

    # Left Shoulder (~142) -> Trough 1 (~128) -> Head (~156) -> Trough 2 (~127) -> Right Shoulder (~141) -> Decline
    s1 = np.linspace(118, 142, 15)  # Rally to Left Shoulder
    s2 = np.linspace(142, 128, 10)  # Valley 1
    s3 = np.linspace(128, 156, 12)  # Rally to Head (Peak)
    s4 = np.linspace(156, 127, 12)  # Valley 2
    s5 = np.linspace(127, 141, 10)  # Rally to Right Shoulder
    s6 = np.linspace(141, 116, 15)  # Breakdown below neckline

    trend = np.concatenate([s1, s2, s3, s4, s5, s6])
    noise = np.random.normal(0, 0.7, len(trend))
    closes = trend + noise

    opens = np.roll(closes, 1)
    opens[0] = closes[0] - 0.5
    highs = np.maximum(opens, closes) + np.random.uniform(0.2, 1.4, len(closes))
    lows = np.minimum(opens, closes) - np.random.uniform(0.2, 1.4, len(closes))

    fig, ax, text_color = create_chart_canvas(1000, 600, dark_theme=True)
    draw_candlesticks(ax, opens, highs, lows, closes)

    ax.set_title(
        "ETH/USDT - 1D | Classic Head and Shoulders Pattern",
        color=text_color,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(-1, len(closes) + 2)
    ax.set_ylim(min(lows) - 5, max(highs) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated Head and Shoulders sample at: {output_path}")


def generate_double_bottom_chart(
    output_path="samples/sample_double_bottom.png",
):
    """Generates a realistic chart with a Double Bottom ('W') pattern."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(88)

    # Downtrend -> Bottom 1 (~102) -> Peak Rebound (~118) -> Bottom 2 (~103) -> Breakout (~130)
    s1 = np.linspace(130, 102, 18)
    s2 = np.linspace(102, 118, 12)
    s3 = np.linspace(118, 103, 12)
    s4 = np.linspace(103, 132, 18)

    trend = np.concatenate([s1, s2, s3, s4])
    noise = np.random.normal(0, 0.7, len(trend))
    closes = trend + noise

    opens = np.roll(closes, 1)
    opens[0] = closes[0] + 0.5
    highs = np.maximum(opens, closes) + np.random.uniform(0.2, 1.2, len(closes))
    lows = np.minimum(opens, closes) - np.random.uniform(0.2, 1.2, len(closes))

    fig, ax, text_color = create_chart_canvas(1000, 600, dark_theme=True)
    draw_candlesticks(ax, opens, highs, lows, closes)

    ax.set_title(
        "AAPL - Daily | Double Bottom ('W') Bullish Reversal",
        color=text_color,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(-1, len(closes) + 2)
    ax.set_ylim(min(lows) - 5, max(highs) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated Double Bottom sample at: {output_path}")


def generate_all_samples():
    generate_double_top_chart()
    generate_head_and_shoulders_chart()
    generate_double_bottom_chart()


if __name__ == "__main__":
    generate_all_samples()
