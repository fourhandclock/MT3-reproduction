"""Training-only translation of frozen right-edge supervision.

The feature grid is unchanged: frame i becomes available at (i + 1) hops.
With a positive delay d, its label is the original label at frame i - d.
Apply this operation to a complete cached recording *before* taking crops.
Original annotation times must still be used for evaluation.
"""

from collections.abc import Mapping

import numpy as np


def delay_targets(targets, delay_frames, config):
    """Return independent target arrays translated later by ``delay_frames``.

    No feature samples or future labels are read. Leading state is inactive;
    other leading heads are zero. A supplied one-dimensional validity mask is
    translated with its labels and has zero validity in the leading region.
    Conflicts (-100), mask values, array lengths and dtypes are preserved.

    This is an exact array translation, not an annotation re-labeler. Gaussian
    boundary support is translated and clipped like other labels, even if its
    center passes the last frame. Already-censored source boundaries remain
    zero; no offset is synthesized to end a note at the recording boundary.
    """
    if isinstance(delay_frames, (bool, np.bool_)) or not isinstance(
            delay_frames, (int, np.integer)) or delay_frames < 0:
        raise ValueError("Target delay must be a nonnegative integer frame count")
    required = {"state", "onset", "offset", "pitch"}
    # NpzFile implements Mapping, as do the normal dict target collections.
    if not isinstance(targets, Mapping) or not required.issubset(targets):
        raise ValueError("Targets must contain state, onset, offset and pitch")
    if set(targets) - required - {"mask"}:
        raise ValueError("Unexpected target array; only an optional mask is supported")
    model = config["model"]
    inactive = int(model["max_fret"]) + 1
    pitch_count = int(model["pitch_max"]) - int(model["pitch_min"]) + 1
    if inactive < 1 or pitch_count < 1:
        raise ValueError("Target class specification must be positive")
    arrays = {key: np.asarray(value) for key, value in targets.items()}
    states = arrays["state"]
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("State targets must have shape [frames, 6]")
    length = len(states)
    if not np.issubdtype(states.dtype, np.integer):
        raise ValueError("State targets must have an integer dtype")
    if not np.iinfo(states.dtype).min <= inactive <= np.iinfo(states.dtype).max:
        raise ValueError("State dtype cannot represent the inactive class")
    if np.any((states != -100) & ((states < 0) | (states > inactive))):
        raise ValueError("State targets contain an unsupported class")
    for key, width in [("onset", 6), ("offset", 6), ("pitch", pitch_count)]:
        value = arrays[key]
        if value.shape != (length, width):
            raise ValueError(f"{key} targets have an inconsistent shape")
        if (not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value) or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError(f"{key} targets must contain finite values in [0, 1]")
    if "mask" in arrays:
        mask = arrays["mask"]
        if (mask.shape != (length,) or np.iscomplexobj(mask)
                or not (np.issubdtype(mask.dtype, np.number)
                        or np.issubdtype(mask.dtype, np.bool_))
                or not np.isfinite(mask).all()
                or np.any((mask < 0) | (mask > 1))):
            raise ValueError("Mask must contain one finite validity value in [0, 1] per frame")
    result = {key: np.full_like(value, inactive) if key == "state"
              else np.zeros_like(value) for key, value in arrays.items()}
    amount = min(int(delay_frames), length)
    if amount < length:
        for key, value in arrays.items():
            result[key][amount:] = value[:length - amount]
    return result
