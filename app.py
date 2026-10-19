import requests
import streamlit as st
import pandas as pd
import numpy as np


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Gold Pattern Explorer",
    layout="wide"
)

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")
st.info(
    "يجلب التطبيق بيانات الذهب تلقائيًا عند الإمكان، مع الاحتفاظ بإمكانية "
    "رفع ملف CSV يدويًا كخيار بديل."
)

API_URL = "https://xaus.com/api/v1/chart"
MIN_HISTORY_MONTHS = 6
MAX_LATEST_CANDLE_AGE_HOURS = 72


# ==================================================
# 2. Data helpers
# ==================================================

def validate_history(dataframe):
    """Normalize OHLC history and reject invalid or insufficient data."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("لم تصل أي شموع صالحة من مصدر البيانات.")

    history = dataframe.copy()
    history.columns = [str(column).strip().lower() for column in history.columns]

    # Accept both CSV names and XAUS API's compact OHLCV names.
    history = history.rename(columns={
        "t": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(
            "الأعمدة المطلوبة غير موجودة: " + ", ".join(missing_columns)
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"], utc=True, errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "volume" in history.columns:
        history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    else:
        history["volume"] = np.nan

    history = history.dropna(subset=["timestamp"] + price_columns)
    history = history.sort_values("timestamp")
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    valid_ohlc = (
        (history["high"] >= history["low"])
        & (history["high"] >= history["open"])
        & (history["high"] >= history["close"])
        & (history["low"] <= history["open"])
        & (history["low"] <= history["close"])
        & (history[price_columns] > 0).all(axis=1)
    )
    history = history.loc[valid_ohlc].reset_index(drop=True)

    if history.empty:
        raise ValueError("لم تبقَ أي شموع صالحة بعد التحقق من الأسعار.")

    # Require at least six calendar months between the first and last candles.
    first_timestamp = history["timestamp"].min()
    last_timestamp = history["timestamp"].max()
    six_months_before_last = last_timestamp - pd.DateOffset(months=MIN_HISTORY_MONTHS)
    if first_timestamp > six_months_before_last:
        raise ValueError(
            "الفترة التاريخية المستلمة أقل من ستة أشهر. "
            f"الفترة المتاحة: {first_timestamp} — {last_timestamp}."
        )

    # The latest candle must not be excessively old. A 72-hour threshold allows
    # normal weekend/holiday gaps but rejects obviously stale automatic data.
    age = pd.Timestamp.now(tz="UTC") - last_timestamp
    if age > pd.Timedelta(hours=MAX_LATEST_CANDLE_AGE_HOURS):
        raise ValueError(
            "آخر شمعة قديمة جدًا للاستخدام التلقائي "
            f"(عمرها التقريبي {age.total_seconds() / 3600:.1f} ساعة)."
        )

    return history


def api_timestamp(value):
    """Parse XAUS timestamps expressed as Unix seconds/milliseconds or ISO text."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return pd.NaT
        unit = "ms" if abs(float(value)) >= 1e12 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_automatic_history():
    """Fetch two years of hourly XAU/USD candles and validate the response."""
    response = requests.get(
        API_URL,
        params={"symbol": "xau", "range": "2y", "interval": "1h"},
        timeout=25,
        headers={"User-Agent": "GoldPatternExplorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    data_state = payload.get("data_state", {}) if isinstance(payload, dict) else {}
    api_status = str(data_state.get("status", "")).lower() if isinstance(data_state, dict) else ""
    if api_status == "unavailable":
        raise ValueError("مصدر البيانات يعلن أن البيانات غير متاحة حاليًا.")
    if api_status == "stale":
        raise ValueError("مصدر البيانات يعلن أن البيانات التاريخية قديمة؛ لم تُستخدم تلقائيًا.")

    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = None
        for key in ("points", "data", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break
        if points is None:
            # Handle a standard chart-response wrapper if the upstream changes shape.
            chart = payload.get("chart")
            if isinstance(chart, dict):
                result = chart.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    timestamps = result[0].get("timestamp", [])
                    quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
                    points = [
                        {
                            "t": timestamps[i],
                            "o": (quote.get("open") or [None] * len(timestamps))[i],
                            "h": (quote.get("high") or [None] * len(timestamps))[i],
                            "l": (quote.get("low") or [None] * len(timestamps))[i],
                            "c": (quote.get("close") or [None] * len(timestamps))[i],
                            "v": (quote.get("volume") or [None] * len(timestamps))[i],
                        }
                        for i in range(len(timestamps))
                    ]
        if points is None:
            raise ValueError("صيغة استجابة مصدر البيانات غير معروفة.")
    else:
        raise ValueError("استجابة مصدر البيانات ليست بصيغة JSON المتوقعة.")

    if not points:
        raise ValueError("مصدر البيانات لم يُرجع أي شموع.")

    raw = pd.DataFrame(points)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    timestamp_column = next(
        (column for column in ("t", "timestamp", "time", "datetime", "date") if column in raw.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError("استجابة مصدر البيانات لا تحتوي على وقت الشمعة.")

    raw["timestamp"] = raw[timestamp_column].map(api_timestamp)
    raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if timestamp_column != "timestamp" and timestamp_column in raw.columns:
        # The parsed timestamp column above is authoritative.
        raw = raw.drop(columns=[timestamp_column])

    history = validate_history(raw)
    return history, data_state


def load_csv_history(uploaded_file):
    """Read an uploaded CSV using the same validation rules as automatic data."""
    csv_data = pd.read_csv(uploaded_file)
    return validate_history(csv_data)


# ==================================================
# 3. Automatic data fetching, with manual fallback
# ==================================================

if "auto_fetch_attempted" not in st.session_state:
    st.session_state["auto_fetch_attempted"] = True
    try:
        with st.spinner("جارٍ جلب بيانات الذهب التاريخية تلقائيًا..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
    except Exception as error:
        st.session_state["auto_history_data"] = None
        st.session_state["auto_data_state"] = {}
        st.session_state["auto_fetch_error"] = str(error)
        st.session_state["source_mode"] = "Upload CSV"

st.header("1. Data Source")

if st.button("🔄 Refresh automatic gold data"):
    try:
        with st.spinner("جارٍ تحديث بيانات الذهب التاريخية..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
        st.success("تم تحديث البيانات التلقائية بنجاح.")
    except Exception as error:
        st.session_state["auto_fetch_error"] = str(error)
        st.warning(
            "تعذّر تحديث البيانات التلقائية. يمكنك متابعة العمل بملف CSV اليدوي. "
            f"التفاصيل: {error}"
        )

source_mode = st.radio(
    "Choose data source",
    options=["Automatic data", "Upload CSV"],
    horizontal=True,
    key="source_mode",
)

history = None

if source_mode == "Automatic data":
    auto_history = st.session_state.get("auto_history_data")
    if isinstance(auto_history, pd.DataFrame) and not auto_history.empty:
        history = auto_history.copy()
        st.success(
            f"تم تحميل {len(history):,} شمعة تاريخية تلقائيًا. "
            "المصدر: XAUS (XAU/USD، إطار الساعة)."
        )
        st.caption(
            "الفترة التاريخية (UTC): "
            f"{history['timestamp'].min()} — {history['timestamp'].max()}"
        )
        state = st.session_state.get("auto_data_state", {})
        if isinstance(state, dict) and state.get("as_of"):
            st.caption(f"حالة/وقت المصدر حسب API: {state.get('status', 'غير محدد')} — {state['as_of']}")
        st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
    else:
        error_message = st.session_state.get("auto_fetch_error")
        st.warning(
            "البيانات التلقائية غير متاحة حاليًا. اختر «Upload CSV» "
            "للعودة إلى الطريقة القديمة، أو اضغط زر التحديث لإعادة المحاولة."
        )
        if error_message:
            st.caption(f"تفاصيل فشل الجلب التلقائي: {error_message}")
else:
    history_file = st.file_uploader(
        "Upload historical gold data (CSV)",
        type=["csv"],
        key="history",
        help="يجب أن يحتوي الملف على timestamp, open, high, low, close."
    )
    if history_file is not None:
        try:
            history = load_csv_history(history_file)
            st.session_state["csv_history_data"] = history
            st.success(f"Loaded {len(history):,} valid candles from CSV.")
            st.caption(
                "الفترة التاريخية (UTC): "
                f"{history['timestamp'].min()} — {history['timestamp'].max()}"
            )
            st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"تعذرت قراءة ملف CSV أو التحقق منه: {error}")
    elif isinstance(st.session_state.get("csv_history_data"), pd.DataFrame):
        history = st.session_state["csv_history_data"].copy()
        st.info("تم الاحتفاظ بملف CSV الذي رفعته سابقًا خلال هذه الجلسة.")

if isinstance(history, pd.DataFrame):
    st.session_state["history_data"] = history.copy()
else:
    st.session_state.pop("history_data", None)


# ==================================================
# 4. Pattern matching helpers
# ==================================================

def normalized_close_shape(frame):
    closes = frame["close"].to_numpy(dtype=float)
    if len(closes) < 3 or not np.isfinite(closes).all() or closes[0] <= 0:
        return None
    return closes / closes[0] - 1.0


def turning_points(values):
    """Return compressed local turning points as (index, H/L, value)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return []
    raw = []
    for i in range(1, len(values) - 1):
        left, mid, right = values[i - 1], values[i], values[i + 1]
        if mid >= left and mid > right:
            raw.append((i, "H", float(mid)))
        elif mid <= left and mid < right:
            raw.append((i, "L", float(mid)))
    compressed = []
    for point in raw:
        if not compressed or compressed[-1][1] != point[1]:
            compressed.append(point)
        else:
            old = compressed[-1]
            if (point[1] == "H" and point[2] > old[2]) or (point[1] == "L" and point[2] < old[2]):
                compressed[-1] = point
    return compressed


def structure_points_compatible(target_pivots, candidate_pivots, pattern_length):
    """Score swing structure softly instead of rejecting candidates for minor pivot differences.

    The broad shape remains the main similarity factor. Different counts of small turns,
    one extra/missing swing, or a slightly shifted turning point lower the structure score
    rather than disqualifying the entire historical window.
    """
    if not target_pivots or not candidate_pivots:
        return True, 25.0

    # Compare the H/L sequences using normalized edit distance. This tolerates an extra
    # small peak/trough while still rewarding candidates with a similar swing order.
    target_types = [p[1] for p in target_pivots]
    candidate_types = [p[1] for p in candidate_pivots]
    rows, cols = len(target_types) + 1, len(candidate_types) + 1
    edit = np.zeros((rows, cols), dtype=float)
    edit[:, 0] = np.arange(rows)
    edit[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = 0.0 if target_types[i - 1] == candidate_types[j - 1] else 1.0
            edit[i, j] = min(
                edit[i - 1, j] + 1.0,
                edit[i, j - 1] + 1.0,
                edit[i - 1, j - 1] + substitution,
            )
    type_similarity = max(0.0, 1.0 - edit[-1, -1] / max(len(target_types), len(candidate_types), 1))

    # Compare the overall pivot path on a common grid, allowing different pivot counts.
    target_pos = np.array([p[0] / max(1, pattern_length - 1) for p in target_pivots])
    candidate_pos = np.array([p[0] / max(1, pattern_length - 1) for p in candidate_pivots])
    target_val = np.array([p[2] for p in target_pivots], dtype=float)
    candidate_val = np.array([p[2] for p in candidate_pivots], dtype=float)
    grid = np.linspace(0.0, 1.0, 9)
    target_path = np.interp(grid, target_pos, target_val, left=target_val[0], right=target_val[-1])
    candidate_path = np.interp(grid, candidate_pos, candidate_val, left=candidate_val[0], right=candidate_val[-1])
    path_error = float(np.mean(np.abs(target_path - candidate_path)))
    path_score = float(np.exp(-12.0 * path_error))

    # Preserve peak-vs-peak and trough-vs-trough rank relationships when the two
    # patterns have comparable counts, but treat disagreements as a penalty, not a gate.
    rank_scores = []
    for kind in ("H", "L"):
        tv = [p[2] for p in target_pivots if p[1] == kind]
        cv = [p[2] for p in candidate_pivots if p[1] == kind]
        n = min(len(tv), len(cv))
        if n >= 2:
            mismatches = 0
            comparisons = 0
            for i in range(n):
                for j in range(i + 1, n):
                    comparisons += 1
                    if np.sign(tv[i] - tv[j]) != np.sign(cv[i] - cv[j]):
                        mismatches += 1
            rank_scores.append(1.0 - mismatches / max(1, comparisons))
        elif n == 1:
            rank_scores.append(0.75)
        # Different peak/trough counts are a soft penalty.
        if max(len(tv), len(cv)) > 0:
            rank_scores.append(min(len(tv), len(cv)) / max(len(tv), len(cv)))

    rank_score = float(np.mean(rank_scores)) if rank_scores else 0.5
    count_similarity = min(len(target_pivots), len(candidate_pivots)) / max(len(target_pivots), len(candidate_pivots))
    structure_score = 100.0 * (
        0.40 * type_similarity + 0.30 * path_score + 0.20 * rank_score + 0.10 * count_similarity
    )
    return True, float(np.clip(structure_score, 0.0, 100.0))


def structure_is_compatible(target_shape, candidate_shape):
    return structure_points_compatible(turning_points(target_shape), turning_points(candidate_shape), len(target_shape))


@st.cache_data(show_spinner=False, max_entries=8)
def precompute_pattern_windows(history, pattern_length):
    """Cache normalized candle windows and their pivots so multi-sample backtests stay practical."""
    count = len(history) - pattern_length + 1
    shapes = np.full((max(0, count), pattern_length), np.nan, dtype=float)
    pivots = []
    closes = history["close"].to_numpy(dtype=float)
    for start in range(max(0, count)):
        window = closes[start:start + pattern_length]
        if len(window) == pattern_length and np.isfinite(window).all() and window[0] > 0:
            shape = window / window[0] - 1.0
            shapes[start] = shape
            pivots.append(turning_points(shape))
        else:
            pivots.append([])
    return shapes, pivots


def compare_patterns(target, candidate):
    target_shape = normalized_close_shape(target)
    candidate_shape = normalized_close_shape(candidate)
    if target_shape is None or candidate_shape is None or len(target_shape) != len(candidate_shape):
        return None
    compatible, structure_score = structure_is_compatible(target_shape, candidate_shape)
    if not compatible:
        return None
    rmse = float(np.sqrt(np.mean((target_shape - candidate_shape) ** 2)))
    shape_score = float(100.0 * np.exp(-25.0 * rmse))
    # Geometry is deliberately the dominant component; structure is a hard gate and a secondary score.
    similarity = 0.85 * shape_score + 0.15 * structure_score
    return {"rmse": rmse, "shape_score": shape_score,
            "structure_score": structure_score, "similarity": float(similarity)}


def next_three_candle_signal(history, candidate_end):
    future = history.iloc[candidate_end:candidate_end + 3]
    if len(future) != 3:
        return None
    green = (future["close"].to_numpy() > future["open"].to_numpy()).all()
    red = (future["close"].to_numpy() < future["open"].to_numpy()).all()
    if green:
        return "BUY"
    if red:
        return "SELL"
    return None


def find_historical_matches(history, target_start, pattern_length, min_similarity=0.0):
    target = history.iloc[target_start:target_start + pattern_length]
    if len(target) != pattern_length:
        return []
    target_shape = normalized_close_shape(target)
    if target_shape is None:
        return []
    all_shapes, all_pivots = precompute_pattern_windows(history, pattern_length)
    last_candidate_start = target_start - pattern_length - 3
    if last_candidate_start < 0:
        return []
    starts = np.arange(last_candidate_start + 1)
    valid = np.isfinite(all_shapes[starts]).all(axis=1)
    starts = starts[valid]
    if not len(starts):
        return []
    # Vectorized shape screen, followed by a tolerant pivot-structure score.
    errors = np.sqrt(np.mean((all_shapes[starts] - target_shape) ** 2, axis=1))
    shape_scores = 100.0 * np.exp(-25.0 * errors)
    possible = (0.85 * shape_scores + 15.0) >= min_similarity
    starts, errors, shape_scores = starts[possible], errors[possible], shape_scores[possible]
    target_pivots = turning_points(target_shape)
    results = []
    for start, rmse, shape_score in zip(starts.tolist(), errors.tolist(), shape_scores.tolist()):
        compatible, structure_score = structure_points_compatible(target_pivots, all_pivots[start], pattern_length)
        if not compatible:
            continue
        similarity = 0.85 * shape_score + 0.15 * structure_score
        if similarity < min_similarity:
            continue
        end = start + pattern_length
        results.append({
            "start": start, "end": end, "similarity": float(similarity),
            "rmse": float(rmse), "shape_score": float(shape_score),
            "structure_score": float(structure_score),
            "signal": next_three_candle_signal(history, end),
            "future": history.iloc[end:end + 3].copy(),
        })
    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results


def calculate_atr(history, index, period=14):
    """ATR using candles no later than index; no future information is used."""
    start = max(1, index - period + 1)
    rows = history.iloc[start:index + 1]
    if rows.empty:
        return np.nan
    previous_close = history["close"].shift(1).iloc[start:index + 1].to_numpy(dtype=float)
    high = rows["high"].to_numpy(dtype=float)
    low = rows["low"].to_numpy(dtype=float)
    true_ranges = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    valid = true_ranges[np.isfinite(true_ranges) & (true_ranges > 0)]
    return float(np.mean(valid)) if len(valid) else np.nan


def simulate_trade(history, entry_index, direction, equity, leverage, min_notional,
                   max_notional, atr_multiplier, cost_per_side, liquidation_floor):
    entry = float(history.iloc[entry_index]["close"])
    atr = calculate_atr(history, entry_index)
    if not np.isfinite(atr) or atr <= 0 or entry <= 0:
        return {"status": "no_atr", "equity_after": equity}

    risk_cash = equity * 0.05
    # ATR provides the initial stop distance. Broker notional limits constrain it so
    # the modeled position remains within the requested 5% risk / 10% target design.
    stop_distance = atr * atr_multiplier
    stop_distance = min(stop_distance, entry * risk_cash / min_notional)
    stop_distance = max(stop_distance, entry * risk_cash / max_notional)
    stop_fraction = stop_distance / entry
    if stop_fraction <= 0:
        return {"status": "invalid_stop", "equity_after": equity}

    notional = risk_cash / stop_fraction
    notional = float(np.clip(notional, min_notional, max_notional))
    margin = notional / leverage
    if equity < margin:
        return {"status": "insufficient_margin", "equity_after": equity,
                "required_margin": margin, "notional": notional}

    if direction == "BUY":
        stop_price = entry - stop_distance
        target_price = entry + 2.0 * stop_distance
    else:
        stop_price = entry + stop_distance
        target_price = entry - 2.0 * stop_distance

    exit_index = None
    exit_price = None
    exit_reason = None
    for j in range(entry_index + 1, len(history)):
        bar = history.iloc[j]
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if direction == "BUY":
            stop_hit = l <= stop_price
            target_hit = h >= target_price
            if stop_hit and target_hit:
                exit_index, exit_price, exit_reason = j, min(o, stop_price), "STOP (both touched; conservative)"
                break
            if stop_hit:
                exit_index, exit_price, exit_reason = j, min(o, stop_price), "STOP"
                break
            if target_hit:
                exit_index, exit_price, exit_reason = j, target_price, "TARGET"
                break
        else:
            stop_hit = h >= stop_price
            target_hit = l <= target_price
            if stop_hit and target_hit:
                exit_index, exit_price, exit_reason = j, max(o, stop_price), "STOP (both touched; conservative)"
                break
            if stop_hit:
                exit_index, exit_price, exit_reason = j, max(o, stop_price), "STOP"
                break
            if target_hit:
                exit_index, exit_price, exit_reason = j, target_price, "TARGET"
                break

        # Conservative stop-out model: close if equity would fall to/below the
        # $0.10 liquidation floor; with OHLC-only bars, use the bar's adverse extreme.
        adverse_price = l if direction == "BUY" else h
        adverse_return = ((adverse_price - entry) / entry) * (1 if direction == "BUY" else -1)
        floating_equity = equity + notional * adverse_return - notional * cost_per_side * 2
        if floating_equity <= liquidation_floor:
            exit_index, exit_price, exit_reason = j, adverse_price, "LIQUIDATION FLOOR"
            break

    if exit_index is None:
        return {"status": "unclosed_at_data_end", "equity_after": equity,
                "entry_index": entry_index, "notional": notional, "margin": margin,
                "entry": entry, "stop": stop_price, "target": target_price}

    direction_sign = 1 if direction == "BUY" else -1
    gross_pnl = notional * ((exit_price - entry) / entry) * direction_sign
    costs = notional * cost_per_side * 2
    net_pnl = gross_pnl - costs
    equity_after = max(0.0, equity + net_pnl)
    status = "closed"
    if exit_reason == "LIQUIDATION FLOOR" or equity_after <= liquidation_floor:
        equity_after = max(0.0, equity_after)
        status = "liquidated"
    return {
        "status": status, "equity_before": equity, "equity_after": equity_after,
        "entry_index": entry_index, "exit_index": exit_index,
        "entry_time": history.iloc[entry_index]["timestamp"],
        "exit_time": history.iloc[exit_index]["timestamp"],
        "direction": direction, "entry": entry, "exit": exit_price,
        "stop": stop_price, "target": target_price, "exit_reason": exit_reason,
        "notional": notional, "margin": margin, "atr": atr,
        "gross_pnl": gross_pnl, "costs": costs, "net_pnl": net_pnl,
        "net_return_pct_equity": (net_pnl / equity * 100) if equity else 0.0,
    }


def run_backtest(history, pattern_length, initial_equity=10.0, max_trades=30,
                 sample_gap_hours=24, similarity_threshold=80.0, leverage=1000.0,
                 min_notional=1000.0, max_notional=1_000_000.0,
                 atr_multiplier=1.5, cost_per_side=0.0001, liquidation_floor=0.10):
    equity = float(initial_equity)
    trades = []
    decisions = []
    # At decision candle i, target pattern ends at i-4, matching the user's four-hour delay.
    i = 2 * pattern_length + 6
    step = max(24, int(sample_gap_hours))
    stop_reason = "انتهت البيانات التاريخية قبل إكمال 30 صفقة."
    while i < len(history) and len(trades) < max_trades:
        if equity <= liquidation_floor:
            stop_reason = f"توقف الاختبار لأن الرصيد وصل إلى حد التصفية ({liquidation_floor:.2f}$)."
            break
        target_start = i - pattern_length - 3
        if target_start < pattern_length + 3:
            i += step
            continue
        matches = find_historical_matches(history, target_start, pattern_length, min_similarity=similarity_threshold)
        best = matches[0] if matches else None
        decision = {"Decision time": history.iloc[i]["timestamp"],
                    "Target start": history.iloc[target_start]["timestamp"],
                    "Best similarity (%)": round(best["similarity"], 2) if best else np.nan,
                    "Signal": best["signal"] if best else None,
                    "Action": "No match > threshold" if best is None else (best["signal"] or "No trade: mixed/doji")}
        decisions.append(decision)
        if best is None or best["signal"] not in ("BUY", "SELL"):
            i += step
            continue

        result = simulate_trade(history, i, best["signal"], equity, leverage, min_notional,
                                max_notional, atr_multiplier, cost_per_side, liquidation_floor)
        if result["status"] == "insufficient_margin":
            stop_reason = (f"توقف الاختبار: الهامش المطلوب ({result.get('required_margin', 0):.2f}$) "
                           f"أكبر من الرصيد المتاح ({equity:.2f}$). تحقق من رافعة الوسيط.")
            break
        if result["status"] == "unclosed_at_data_end":
            stop_reason = "وصل الاختبار إلى نهاية البيانات قبل أن يلمس السعر الهدف أو الوقف؛ لم تُحتسب الصفقة غير المغلقة."
            break
        if result["status"] in ("no_atr", "invalid_stop"):
            i += step
            continue

        result["trade_number"] = len(trades) + 1
        result["similarity"] = best["similarity"]
        result["match_start_time"] = history.iloc[best["start"]]["timestamp"]
        result["match_end_time"] = history.iloc[best["end"] - 1]["timestamp"]
        trades.append(result)
        equity = result["equity_after"]
        if result["status"] == "liquidated" or equity <= liquidation_floor:
            stop_reason = f"توقف الاختبار لأن الرصيد وصل إلى حد التصفية ({liquidation_floor:.2f}$)."
            break
        # Do not overlap trades; also preserve the 24h/48h gap between decision samples.
        i = max(i + step, int(result["exit_index"]) + step)

    if len(trades) >= max_trades:
        stop_reason = f"اكتمل الاختبار بعد إغلاق {max_trades} صفقة."
    elif len(history) <= i:
        stop_reason = "انتهت البيانات التاريخية قبل إكمال العدد المطلوب من الصفقات."
    return trades, decisions, equity, stop_reason


# ==================================================
# 5. Historical pattern matching
# ==================================================

st.divider()
st.header("2. Historical Matching")

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات وحمّل البيانات قبل بدء البحث عن الأنماط.")
else:
    history = st.session_state["history_data"].copy()
    pattern_length = st.selectbox(
        "Pattern length (hourly candles)", [72, 96, 144, 216], index=2,
        key="matching_pattern_length"
    )
    if len(history) < 2 * pattern_length + 7:
        st.warning(f"تحتاج إلى بيانات أكثر للبحث عن نمط بطول {pattern_length} شمعة.")
    else:
        default_start = len(history) - pattern_length - 4
        max_start = len(history) - pattern_length - 4
        target_start = int(st.number_input(
            "Target starting candle index (latest usable target by default)",
            min_value=pattern_length + 3, max_value=max_start,
            value=default_start, step=1, key=f"matching_target_start_{pattern_length}"
        ))
        target = history.iloc[target_start:target_start + pattern_length].copy()
        st.write("Target period (UTC):", str(target["timestamp"].iloc[0]), "—", str(target["timestamp"].iloc[-1]))
        st.caption("تُترك أربع شموع ساعة بعد نهاية العينة قبل نقطة القرار. اختلاف عدد القمم والقيعان أو وجود انعطاف صغير إضافي يخفض درجة البنية بدل استبعاد النمط تلقائيًا.")
        st.line_chart(target.set_index("timestamp")["close"])
        with st.spinner("جارٍ البحث عن الأنماط التاريخية المتوافقة هندسيًا..."):
            matches = find_historical_matches(history, target_start, pattern_length, min_similarity=0.0)
        if not matches:
            st.warning("لم يُعثر على أنماط مطابقة في هذه العينة. أصبح شرط بنية القمم والقيعان أكثر مرونة؛ جرّب عينة أخرى أو تحقق من طول النمط والبيانات المتاحة.")
        else:
            st.success(f"عُثر على {len(matches):,} نمط متوافق بنيويًا.")
            top_n = st.slider("Number of matches to display", 1, min(20, len(matches)), min(10, len(matches)))
            rows = []
            for rank, item in enumerate(matches[:top_n], 1):
                matched = history.iloc[item["start"]:item["end"]]
                rows.append({"Rank": rank, "Similarity (%)": round(item["similarity"], 2),
                             "Shape score (%)": round(item["shape_score"], 2),
                             "Structure score (%)": round(item["structure_score"], 2),
                             "Three-candle signal": item["signal"] or "No trade",
                             "Start (UTC)": str(matched["timestamp"].iloc[0]),
                             "End (UTC)": str(matched["timestamp"].iloc[-1])})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("نسبة التشابه مقياس هندسي وليست احتمالًا لربح الصفقة. الشكل الهندسي هو العامل الرئيسي، وبنية القمم والقيعان عامل مرن يخفض الدرجة عند الاختلاف بدل أن يكون شرط رفض صارمًا.")
            selected_rank = st.selectbox("Inspect a historical match", list(range(1, top_n + 1)),
                                         format_func=lambda rank: f"Match {rank}")
            selected = matches[selected_rank - 1]
            matched = history.iloc[selected["start"]:selected["end"]].copy()
            st.write(f"Similarity: {selected['similarity']:.2f}% — signal: {selected['signal'] or 'No trade'}")
            st.subheader("Matched historical pattern")
            st.line_chart(matched.set_index("timestamp")["close"])
            st.subheader("Three candles after the matched pattern")
            st.dataframe(selected["future"][["timestamp", "open", "high", "low", "close"]],
                         use_container_width=True, hide_index=True)


# ==================================================
# 6. Walk-forward backtesting
# ==================================================

st.divider()
st.header("3. Backtesting — 30 closed trades")
st.write("يختبر التطبيق العينات التاريخية بالتتابع، ويستخدم فقط المعلومات المتاحة حتى شمعة القرار. لا يفتح صفقة جديدة قبل إغلاق السابقة.")

if "history_data" not in st.session_state:
    st.info("حمّل بيانات الذهب أولًا لتشغيل الاختبار.")
else:
    history = st.session_state["history_data"].copy()
    with st.expander("Backtest settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            bt_length = st.selectbox("Pattern length", [72, 96, 144, 216], index=2, key="bt_length")
            initial_equity = st.number_input("Initial capital ($)", min_value=1.0, value=10.0, step=1.0, key="bt_equity")
            sample_gap = st.selectbox("Minimum gap between decision samples", [24, 48], index=0, format_func=lambda x: f"{x} hours", key="bt_gap")
        with col2:
            threshold = st.slider("Minimum similarity (%)", 80, 99, 80, key="bt_threshold")
            leverage = st.select_slider("Leverage (assumption; change to match broker)", options=[20, 50, 100, 200, 500, 1000], value=1000, key="bt_leverage")
            atr_multiplier = st.select_slider("ATR stop multiplier", options=[1.0, 1.25, 1.5, 1.75, 2.0], value=1.5, key="bt_atr_mult")
        with col3:
            min_notional = st.number_input("Minimum position notional ($)", min_value=1000.0, value=1000.0, step=1000.0, key="bt_min_notional")
            max_notional = st.number_input("Maximum position notional ($)", min_value=1000.0, value=1_000_000.0, step=10000.0, key="bt_max_notional")
            cost_bps = st.number_input("Estimated cost per side (basis points of notional)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key="bt_cost_bps")
    st.caption("افتراضات المحاكاة: حجم المركز بين 1,000$ و1,000,000$؛ الرافعة الافتراضية 1000x لأنها غير معروفة من رسالتك ويمكن تغييرها. ATR(14) × المضاعف يحدد المسافة الأساسية للوقف، وتُقيّد المسافة للحفاظ على مخاطرة مخططة 5% وهدف 10% قبل التكاليف. التكاليف والفجوات السعرية قد تجعل النتيجة الفعلية مختلفة.")
    if max_notional < min_notional:
        st.error("الحد الأقصى لحجم المركز يجب أن يكون أكبر من أو مساويًا للحد الأدنى.")
    elif st.button("▶ Run 30-trade backtest", type="primary", key="run_backtest"):
        if len(history) < 2 * bt_length + 100:
            st.error("البيانات التاريخية غير كافية لطول النمط المختار. جرّب طولًا أصغر أو بيانات أقدم.")
        else:
            with st.spinner("يجري اختبار العينات التاريخية والصفقات..."):
                trades, decisions, final_equity, stop_reason = run_backtest(
                    history, pattern_length=int(bt_length), initial_equity=float(initial_equity),
                    max_trades=30, sample_gap_hours=int(sample_gap), similarity_threshold=float(threshold),
                    leverage=float(leverage), min_notional=float(min_notional), max_notional=float(max_notional),
                    atr_multiplier=float(atr_multiplier), cost_per_side=float(cost_bps) / 10000.0,
                    liquidation_floor=0.10
                )
            st.session_state["last_backtest_result"] = {
                "trades": trades, "decisions": decisions, "final_equity": final_equity,
                "stop_reason": stop_reason, "initial_equity": float(initial_equity),
            }

    result = st.session_state.get("last_backtest_result")
    if result:
        trades = result["trades"]
        initial = result["initial_equity"]
        final = result["final_equity"]
        count = len(trades)
        wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("net_pnl", 0) <= 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial capital", f"${initial:.2f}")
        c2.metric("Final capital", f"${final:.2f}", f"{final-initial:+.2f}$")
        c3.metric("Closed trades", f"{count}/30")
        c4.metric("Win rate", f"{(wins/count*100 if count else 0):.1f}%")
        st.info(result["stop_reason"])
        st.write(f"Winning trades: {wins} · Losing/non-profitable trades: {losses} · Net return: {(final/initial-1)*100 if initial else 0:.2f}%")
        if trades:
            table = []
            for t in trades:
                table.append({
                    "#": t["trade_number"], "Direction": t["direction"],
                    "Similarity (%)": round(t["similarity"], 2),
                    "Entry time (UTC)": str(t["entry_time"]), "Exit time (UTC)": str(t["exit_time"]),
                    "Entry": round(t["entry"], 3), "Exit": round(t["exit"], 3),
                    "Stop": round(t["stop"], 3), "Target": round(t["target"], 3),
                    "Position notional ($)": round(t["notional"], 2),
                    "Margin ($)": round(t["margin"], 2), "Gross P/L ($)": round(t["gross_pnl"], 3),
                    "Costs ($)": round(t["costs"], 3), "Net P/L ($)": round(t["net_pnl"], 3),
                    "Balance after ($)": round(t["equity_after"], 3), "Exit reason": t["exit_reason"]
                })
            st.subheader("Closed trade log")
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            equity_points = [{"Trade": 0, "Balance ($)": initial}]
            equity_points.extend({"Trade": t["trade_number"], "Balance ($)": t["equity_after"]} for t in trades)
            st.subheader("Account balance progression")
            st.line_chart(pd.DataFrame(equity_points).set_index("Trade")["Balance ($)"])
        if result["decisions"]:
            with st.expander("Decision samples, including those with no trade"):
                st.dataframe(pd.DataFrame(result["decisions"]), use_container_width=True, hide_index=True)
        st.warning("هذا اختبار تاريخي افتراضي وليس ضمانًا للربح أو محاكاة كاملة لشروط الوسيط. الرافعة الفعلية، السبريد، الانزلاق، التمويل، وقواعد التصفية تختلف من وسيط لآخر. إذا لمس الهدف والوقف داخل الشمعة نفسها، تُحسب النتيجة تحفظيًا على الوقف.")
