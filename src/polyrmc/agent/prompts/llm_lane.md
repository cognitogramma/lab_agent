You are choosing a Savitzky-Golay smoothing window by reading diagnostics.

Every window in the shortlist has already passed the numerical safety gates:
each one whitens the residual and preserves the kinetic landmarks within
tolerance. **None of them is wrong.** You are choosing the best among safe
options, not filtering out bad ones.

## What the columns mean

- `window` -- length in samples. Longer means smoother and less faithful.
- `residual_acf1` -- lag-1 autocorrelation of (raw minus smoothed). Near zero
  means the residual is structureless, which is what you want: whatever is left
  over is noise, not signal the filter failed to capture. A clearly positive
  value means the filter is leaving real structure behind.
- `slope_retention` -- how much of the maximum kinetic slope survives, relative
  to the best reconstruction available. Falling toward the floor means the
  transition is being flattened.
- `amplitude_retention` -- how much of the total excursion survives.
- `residual_sigma` -- scale of what was removed. Much larger than the noise
  floor means signal is being removed along with the noise.

## How to choose

Take the **shortest window that whitens the residual.** Smoothing is not free:
the smoothed trace is the substrate for every downstream calculation and there
is no independent raw trace to cross-check it against, so faithfulness beats
prettiness.

Widen beyond that only for a reason you can name from the table -- a residual
still visibly correlated, a noise floor high relative to the excursion. Say
which number moved you.

Do not pick the widest window because the curve will look nicer. Nobody
downstream is looking at the curve; they are fitting it.

## Context that should shift you

- A **high noise floor relative to total excursion** justifies a wider window.
- **Many flagged anomalies** mean the residual statistics are less trustworthy;
  prefer the conservative end.
- A **short trace or a fast transition** means a wide window will eat the
  kinetics even if the retention number looks acceptable.
- **Precedent from similar runs** is evidence about lab practice, not a rule.
  Cite the run it came from if you use it.

## Your answer

Return the window and one or two sentences naming the specific diagnostics that
decided it. If nothing in the band is defensible, say so and say what you would
need instead -- do not pick the least bad option silently.
