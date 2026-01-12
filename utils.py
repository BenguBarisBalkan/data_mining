import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import Union

@dataclass
class XABCDFound:
    X: int
    A: int
    B: int
    C: int
    D: int # Index of last point in pattern, the entry is on the close of D
    error: float # Error found
    name: str
    bull: bool

def plot_pattern(ohlc: pd.DataFrame, pat: Union[XABCDFound, dict, tuple, list], pad=3, ax=None, show=True, show_lines=True, show_ratios=True, post_bars=0, highlight_interval=False, debug=False):
    """Plot a found XABCD pattern.

    Args:
        ohlc: DataFrame with OHLC data indexed by datetime or integer index.
        pat: Either an `XABCDFound` instance, or a dict/tuple/list containing
             at least X,A,B,C,D (optionally name,bull,error).
        pad: Number of rows to include before X and after D for context.
        ax: Optional matplotlib Axes to draw onto. If None, a new figure/axes is created.
        show: If True, call `plt.show()`; otherwise return the fig and ax for embedding.

    Returns:
        (fig, ax) tuple.
    """

    # Coerce dict/tuple/list into XABCDFound if needed
    pred_label = None
    pred_prob = None
    if not isinstance(pat, XABCDFound):
        if isinstance(pat, dict):
            try:
                X = int(pat['X']); A = int(pat['A']); B = int(pat['B']); C = int(pat['C']); D = int(pat['D'])
                name = str(pat.get('name', ''))
                bull = bool(pat.get('bull', True))
                error = float(pat.get('error', 0.0))
                pred_label = pat.get('pred', None)
                pred_prob = pat.get('pred_prob', None)
            except Exception as e:
                raise ValueError('dict `pat` must contain keys X,A,B,C,D') from e
        elif isinstance(pat, (tuple, list)) and len(pat) >= 5:
            try:
                X, A, B, C, D = pat[:5]
                name = pat[5] if len(pat) > 5 else ''
                bull = pat[6] if len(pat) > 6 else True
                error = pat[7] if len(pat) > 7 else 0.0
                # optional prediction values in tuple/list
                pred_label = pat[8] if len(pat) > 8 else None
                pred_prob = pat[9] if len(pat) > 9 else None
                X = int(X); A = int(A); B = int(B); C = int(C); D = int(D)
                name = str(name)
                bull = bool(bull)
                error = float(error)
            except Exception as e:
                raise ValueError('tuple/list `pat` must contain at least 5 integers (X,A,B,C,D)') from e
        else:
            raise ValueError('`pat` must be XABCDFound or dict/tuple/list with X,A,B,C,D')

        pat = XABCDFound(X=X, A=A, B=B, C=C, D=D, error=error, name=name, bull=bull)

    idx = ohlc.index

    # Compute a data window (slice) that is guaranteed to contain the pattern
    # and is expressed as positional indices into `ohlc` so we can build
    # integer-based x-coordinates for mplfinance that are always valid.
    # Strategy:
    #  - If pattern labels (pat.X, pat.D) exist in `ohlc.index`, use their
    #    positional locations (get_loc) and slice +/- pad around those locations.
    #  - Otherwise assume pat.* are integer positional indices into the full
    #    `ohlc` DataFrame and slice using those positions.
    # Record `slice_start_pos` which is the absolute positional index in `ohlc`
    # corresponding to the first row of `data` (used to convert between label
    # values and positional offsets).
    try:
        loc_X = ohlc.index.get_loc(pat.X)
        loc_D = ohlc.index.get_loc(pat.D)
        slice_start_pos = max(0, loc_X - pad)
        slice_end_pos = min(len(ohlc) - 1, loc_D + pad + int(post_bars))
    except Exception:
        # pattern indices are not index labels; treat them as integer positional
        # indices into the full dataframe
        slice_start_pos = max(0, int(pat.X) - pad)
        slice_end_pos = min(len(ohlc) - 1, int(pat.D) + pad + int(post_bars))

    data = ohlc.iloc[slice_start_pos: slice_end_pos + 1]

    plt.style.use('dark_background')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # use matplotlib date locators/formatters for nice x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    # Use values relative to the sliced `data` so alines coordinates are valid for mplfinance
    if pat.bull:
        s1_data = data['low'].to_numpy()
        s2_data = data['high'].to_numpy()
    else:
        s2_data = data['low'].to_numpy()
        s1_data = data['high'].to_numpy()

    # determine integer positions inside `data` (0-based). We use slice_start_pos
    # to convert absolute positional labels into offsets inside `data` when
    # necessary.
    def pos_of(label):
        # If label exists as an index label in `data`, use its positional location
        if label in data.index:
            return data.index.get_loc(label)
        # Otherwise assume `label` is an absolute integer positional index into
        # the full `ohlc` and convert to an offset relative to `slice_start_pos`.
        p = int(label) - slice_start_pos
        if p < 0 or p >= len(data):
            raise IndexError(f"pattern index {label} not in displayed data window")
        return p

    xX = pos_of(pat.X)
    xA = pos_of(pat.A)
    xB = pos_of(pat.B)
    xC = pos_of(pat.C)
    xD = pos_of(pat.D)

    # Prepare a DataFrame with a DatetimeIndex for mplfinance plotting.
    data_plot = data.copy()
    # If the index is not datetime, prefer the 'date' column if present
    if not isinstance(data_plot.index, pd.DatetimeIndex):
        if 'date' in data_plot.columns:
            data_plot.index = pd.to_datetime(data_plot['date'])
        else:
            # fallback: create a dummy datetime index so mplfinance is happy
            data_plot.index = pd.date_range('2000-01-01', periods=len(data_plot), freq='D')

    # Recompute s1_data/s2_data from data_plot to ensure alignment with plotted candles
    if pat.bull:
        s1_data = data_plot['low'].to_numpy()
        s2_data = data_plot['high'].to_numpy()
        s1_col = 'low'
        s2_col = 'high'
    else:
        s2_data = data_plot['low'].to_numpy()
        s1_data = data_plot['high'].to_numpy()
        s1_col = 'high'
        s2_col = 'low'

    # Helper to read exact price from the plotted dataframe (avoids array/indexing mismatch)
    def price_at(pos, which):
        # pos is integer offset into data_plot
        col = s1_col if which == 's1' else s2_col
        try:
            return float(data_plot.iloc[pos][col])
        except Exception:
            # fallback to arrays
            return s1_data[pos] if which == 's1' else s2_data[pos]

    # Build lines using datetime x-coordinates so they align with the plotted candles
    l0 = [(data_plot.index[xX], price_at(xX,'s1')), (data_plot.index[xA], price_at(xA,'s2'))]
    l1 = [(data_plot.index[xA], price_at(xA,'s2')), (data_plot.index[xB], price_at(xB,'s1'))]
    l2 = [(data_plot.index[xB], price_at(xB,'s1')), (data_plot.index[xC], price_at(xC,'s2'))]
    l3 = [(data_plot.index[xC], price_at(xC,'s2')), (data_plot.index[xD], price_at(xD,'s1'))]

    # Connecting lines
    l4 = [(data_plot.index[xA], price_at(xA,'s2')), (data_plot.index[xC], price_at(xC,'s2'))]
    l5 = [(data_plot.index[xB], price_at(xB,'s1')), (data_plot.index[xD], price_at(xD,'s1'))]
    l6 = [(data_plot.index[xX], price_at(xX,'s1')), (data_plot.index[xB], price_at(xB,'s1'))]
    l7 = [(data_plot.index[xX], price_at(xX,'s1')), (data_plot.index[xD], price_at(xD,'s1'))]

    # Draw candles manually with matplotlib primitives (more robust than relying on mplfinance style here)
    dates = data_plot.index.to_pydatetime()
    xnum = mdates.date2num(dates)
    if len(xnum) > 1:
        width = float(np.median(np.diff(xnum))) * 0.6
    else:
        width = 0.6

    opens = data_plot['open'].to_numpy()
    highs = data_plot['high'].to_numpy()
    lows = data_plot['low'].to_numpy()
    closes = data_plot['close'].to_numpy()

    # draw wicks
    for xi, low, high in zip(dates, lows, highs):
        ax.vlines(xi, low, high, color='white', linewidth=0.8, alpha=0.9, zorder=2)

    # draw bodies using bar so they show on dark background
    for xi, o, c in zip(dates, opens, closes):
        color = 'lime' if c >= o else 'crimson'
        ax.bar(xi, c - o, width=width, bottom=o, color=color, edgecolor='white', align='center', linewidth=0.6, alpha=0.9, zorder=2)

    # Define corner/color mapping and point_map unconditionally so debug can inspect them
    corner_color_map = {'X': 'yellow', 'A': 'magenta', 'B': 'orange', 'C': 'magenta', 'D': 'yellow'}
    point_map = {'X': ('s1', xX), 'A': ('s2', xA), 'B': ('s1', xB), 'C': ('s2', xC), 'D': ('s1', xD)}

    if show_lines:
        # Draw pattern lines manually with matplotlib using datetime x coords
        lines = [l0, l1, l2, l3, l4, l5, l6, l7]
        colors = ['w', 'w', 'w', 'w', 'b', 'b', 'b', 'b']
        for (p0, p1), c in zip(lines, colors):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, linewidth=1.5, zorder=5)

        # Draw small corner markers so it's obvious whether the line endpoints align with price
        for label, (series_name, pos) in point_map.items():
            try:
                val = s1_data[pos] if series_name == 's1' else s2_data[pos]
                dt = data_plot.index[pos]
                ax.scatter([dt], [val], s=80, facecolor='none', edgecolor=corner_color_map[label], linewidth=1.5, zorder=9)
            except Exception:
                # if any mapping fails, skip marker drawing
                pass

    # Highlight X->D interval only (if requested), but keep x-limits wide enough to show post_bars
    if highlight_interval:
        try:
            # Highlight only X -> D (pattern window)
            start_dt = data_plot.index[xX]
            end_pos_highlight = xD
            end_dt_highlight = data_plot.index[end_pos_highlight]
            # compute a small margin using the typical step between dates
            dates = data_plot.index.to_pydatetime()
            xnum = mdates.date2num(dates)
            if len(xnum) > 1:
                step = float(np.median(np.diff(xnum)))
            else:
                step = 1.0
            # add half-step margin on both sides for the highlighted rectangle
            left_h = mdates.num2date(mdates.date2num(start_dt) - 0.5 * step)
            right_h = mdates.num2date(mdates.date2num(end_dt_highlight) + 0.5 * step)
            ax.axvspan(left_h, right_h, color='yellow', alpha=0.12, zorder=0)
            # Set x-limits so the viewer sees the interval and also the following post_bars
            end_pos_xlim = min(len(data_plot)-1, xD + int(post_bars))
            end_dt_xlim = data_plot.index[end_pos_xlim]
            left_xlim = mdates.num2date(mdates.date2num(data_plot.index[0]) - 0.5 * step)
            right_xlim = mdates.num2date(mdates.date2num(end_dt_xlim) + 0.5 * step)
            ax.set_xlim(left_xlim, right_xlim)
            if debug:
                print(f"DEBUG highlight_interval: start_dt={start_dt}, end_dt_highlight={end_dt_highlight}, left_h={left_h}, right_h={right_h}, xD={xD}, post_bars={post_bars}, end_pos_highlight={end_pos_highlight}, end_pos_xlim={end_pos_xlim}, left_xlim={left_xlim}, right_xlim={right_xlim}")
        except Exception as e:
            if debug:
                print('DEBUG highlight_interval exception:', e)
            pass

    # If debug is enabled, print numeric coordinates for each point to help trace mismatches
    if debug:
        coords = {}
        for label, (series_name, pos) in point_map.items():
            try:
                val = s1_data[pos] if series_name == 's1' else s2_data[pos]
                coords[label] = (data_plot.index[pos], float(val))
            except Exception:
                coords[label] = None
        print('DEBUG plot_pattern coords:', coords)

    
    # Text (use data-relative arrays and positions)
    if show_ratios:
        # protect against division by zero
        eps = 1e-12
        xa_ab =  abs(s2_data[xA] - s1_data[xB]) / max(abs(s1_data[xX] - s2_data[xA]), eps)
        ab_bc =  abs(s1_data[xB] - s2_data[xC]) / max(abs(s2_data[xA] - s1_data[xB]), eps)
        bc_cd =  abs(s2_data[xC] - s1_data[xD]) / max(abs(s1_data[xB] - s2_data[xC]), eps)
        xa_ad =  abs(s2_data[xA] - s1_data[xD]) / max(abs(s1_data[xX] - s2_data[xA]), eps)

        # Helper to format/clip unreasonable values (avoid huge numbers from tiny denominators)
        def fmt_val(v):
            try:
                if not np.isfinite(v) or abs(v) > 100:
                    return '—'
                return f"{v:.3f}"
            except Exception:
                return '—'

        s_xa_ab = fmt_val(xa_ab)
        s_ab_bc = fmt_val(ab_bc)
        s_bc_cd = fmt_val(bc_cd)
        s_xa_ad = fmt_val(xa_ad)

        # place numeric labels using datetime x coordinates on the plotted data
        # use a slightly smaller font and bbox to keep them legible
        fontsize_label = 'large'
        bbox = dict(facecolor='black', alpha=0.3, pad=2)
        ax.text(data_plot.index[(xX + xB)//2], (s1_data[xX] + s1_data[xB]) / 2 , s_xa_ab, color='orange', fontsize=fontsize_label, bbox=bbox, zorder=6)
        ax.text(data_plot.index[(xA + xC)//2], (s2_data[xA] + s2_data[xC]) / 2 , s_ab_bc, color='orange', fontsize=fontsize_label, bbox=bbox, zorder=6)
        ax.text(data_plot.index[(xB + xD)//2], (s1_data[xB] + s1_data[xD]) / 2 , s_bc_cd, color='orange', fontsize=fontsize_label, bbox=bbox, zorder=6)
        ax.text(data_plot.index[(xX + xD)//2], (s1_data[xX] + s1_data[xD]) / 2 , s_xa_ad, color='orange', fontsize=fontsize_label, bbox=bbox, zorder=6)
    desc_string = pat.name
    # Do not display the numeric error in the plot (user requested)
    # Create title text showing ground truth and prediction (if available)
    if pred_label is not None:
        try:
            if pred_prob is None:
                pred_text = f"Pred: {pred_label}"
                title_text = f"GT: {pat.name} | {pred_text}"
            else:
                pred_text = f"{pred_label} ({float(pred_prob):.3f})"
                title_text = f"GT: {pat.name} | Pred: {pred_text}"
        except Exception:
            pred_text = f"Pred: {pred_label}"
            title_text = f"GT: {pat.name} | {pred_text}"
    else:
        pred_text = None
        title_text = f"GT: {pat.name}"
    # Apply title to the axes
    try:
        ax.set_title(title_text, color='white', fontsize='x-large')
    except Exception:
        pass

    if pat.bull:
        plt_price = data['high'].max() - 0.05 * (data['high'].max() - data['low'].min())
    else:
        plt_price = data['low'].min() + 0.05 * (data['high'].max() - data['low'].min())

    # anchor description at left-most index of plotted data
    ax.text(data_plot.index[0], plt_price , desc_string, color='yellow', fontsize='x-large', zorder=6)

    # show prediction tag at the right-top if available
    if pred_text is not None:
        pred_color = 'lime' if str(pred_label).lower().startswith('bull') else 'cyan'
        ax.text(data_plot.index[-1], data['high'].max(), pred_text, color=pred_color, fontsize='large', ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5), zorder=7)

    # ensure x-limits and y-limits show the entire slice with margin
    if not highlight_interval:
        ax.set_xlim(data_plot.index[0], data_plot.index[-1])
    y_min = data['low'].min(); y_max = data['high'].max()
    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    if show:
        plt.show()
    return fig, ax