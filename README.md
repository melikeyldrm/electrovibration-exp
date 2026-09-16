# electrovibration-exp

This document summarizes how the electrovibration detection-threshold
experiment is set up, how to run it, which parameters can be configured,
and where recorded data ends up.

## 1. What the System Does

The participant slides a finger across a touchscreen through two
consecutive intervals; one interval carries an electrovibration stimulus,
the other does not (a 2-alternative forced-choice / 2AFC design). The
participant reports which interval felt different. An adaptive staircase
adjusts the stimulus voltage from trial to trial based on correct/incorrect
responses, converging on the participant's detection threshold.

At the same time, two ATI Nano17 force sensors record the normal force the
participant applies to the screen, along with finger position. This lets
every trial be reviewed afterward for grip force, timing, and touch
trajectory alongside the psychophysical response.

The protocol currently running lives under
`experiments/exp01_2afc_threshold/`. Future experiments will be added as
sibling folders under `experiments/`.

## 2. Hardware

A real session involves the following physical devices:

- **Touchscreen** — 3M SCT3250, the participant-facing display the
  electrovibration stimulus is rendered on.
- **Data acquisition card** — NI PCIe-6321. Carries the stimulus analog
  output and the amplifier's monitor input, plus the raw gauge voltages
  from both force sensors. Device names are configurable (`--daq-fs1`,
  `--daq-fs2`, `--daq-stim`), defaulting to `Dev1`/`Dev3`/`Dev2`. See
  [`src/evexp/hardware/nidaq.py`](src/evexp/hardware/nidaq.py) for the
  channel wiring.
- **High-voltage amplifier** — Tabor Electronics 9200A. Amplifies the
  DAQ's stimulus output to the voltage actually delivered to the
  touchscreen.
- **Force/torque sensors** — two ATI Nano17 Titanium 6-axis sensors,
  mounted under the same plate, read through the DAQ as raw gauge voltages
  (6 channels each) and converted to newtons via the calibration matrices
  (`GAIN_FS1`, `GAIN_FS2`) in
  [`src/evexp/hardware/force.py`](src/evexp/hardware/force.py). If a
  sensor is swapped or re-calibrated, update the matrices there.
- **Infrared position sensor** — Neonode NNAMC2300PCEV. Tracks finger
  position on the touchscreen over HID, independent of the DAQ. Enabled
  with `--neonode`; without it, the mouse pointer is used instead. See
  [`src/evexp/hardware/neonode.py`](src/evexp/hardware/neonode.py).

Without `--real-daq`/`--neonode`, `run_gui.py` runs entirely against mocked
devices, so development doesn't require any of this hardware to be
attached.

## 3. Installation

Requires Python 3.9+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -e .
```

`nidaqmx` and `hidapi` are installed as regular dependencies, but they're
only exercised when using real hardware (`--real-daq` / `--neonode`);
everything else runs against mocked devices without needing them.

## 4. Running It

### 4.1 Quick check (no hardware, no GUI)

To verify the trial/staircase/logging pipeline works end to end without any
hardware attached:

```bash
python experiments/exp01_2afc_threshold/run.py
```

This writes a CSV and a convergence plot (PNG) to `data/`.

### 4.2 Real / interactive session

The actual experiment runner, with a participant-facing window and a
separate control window for the experimenter:

```bash
python experiments/exp01_2afc_threshold/run_gui.py
```

By default this uses mocked DAQ hardware and the mouse for finger position,
so it runs on a laptop with nothing plugged in. Useful command-line flags:

| Flag | Effect |
|---|---|
| `--real-daq` | Use real (or NI MAX-simulated) NI-DAQmx cards instead of the mock devices, for both acquisition and stimulus generation. |
| `--daq-fs1`, `--daq-fs2`, `--daq-stim` | NI-DAQmx device names for the two force-sensor cards and the stimulus/monitor card (defaults: `Dev1`, `Dev3`, `Dev2`). Only relevant with `--real-daq`. |
| `--monitor-channel` | Analog input channel read as "current" (default `ai0`); reads whichever channel is currently connected. |
| `--neonode` | Track finger position with the Neonode IR sensor instead of the mouse; falls back to the mouse if the sensor can't be opened. |
| `--list-screens` | Print index/geometry of every connected monitor, then exit — use this to find `display.participant_screen_index` in the config before a real session. |

Both Ctrl+C and closing the participant window trigger a clean shutdown
(DAQ tasks closed, stats printed, an in-progress trial not silently
dropped).

## 5. Experimental Protocol

Each trial follows the same 2AFC sequence:

1. The participant places their finger at the starting position and waits
   out `timing.pre_interval_wait_s`.
2. Two consecutive intervals are presented, separated by a gap; the
   participant slides their finger across the screen at the selected speed
   condition for each.
3. Exactly one of the two intervals carries the electrovibration stimulus
   (which one is randomized per trial); the other is a silent control.
4. The participant reports which interval felt different.
5. The response is scored and, for a valid, non-training trial, fed to the
   staircase, which updates the stimulus level per the configured rule.
6. Force and finger-speed measurements from the trial are checked against
   the `validity` tolerances to decide whether the trial counts.

A fixed number of training trials (`training.n_trials`, at a fixed
`training.voltage`) run before the measured session and never update the
staircase, regardless of validity.

## 6. Where Key Settings Are Read / Set

All session-specific settings live in a single file:
[`config/experiment.yaml`](config/experiment.yaml). Notable sections:

- **`display`** — Physical dimensions of the participant screen (width/
  height in mm) and resolution. Stimulus speed and travel distance are
  computed from this, so it's critical. Currently calibrated for a dev
  laptop; must be re-measured against the real touchscreen before a real
  session (see the `TODO` note in the file).
- **`staircase`** — Staircase rule (3-down/1-up), starting value, step
  sizes, the safety ceiling (`max_value`, which the DAQ output must never
  exceed), and the stopping criterion (number of reversals). The staircase
  itself steps in dB; the stimulus is still stored and applied as a linear
  voltage, so `step_sizes: [5.0, 1.0]` means 5 dB / 1 dB steps, not volts.
- **`timing`** — Pre-interval wait time, gap between the two intervals,
  cursor speed and travel distance (interval duration is derived from
  these).
- **`force`** — Target force, feedback color band, and the force source
  (`source: nano17` for the real sensors, `simulated` for a fake signal,
  `mouse_y` for development via mouse control).
- **`validity`** — How far force or speed can drift from target before a
  trial is discarded and re-presented at the same level. Invalid trials do
  not update the staircase; training trials are never discarded, only
  flagged.
- **`acquisition`** — DAQ sample rate and the in-memory ring buffer length.
- **`training`** — Number of practice trials before the real measurement,
  and their voltage.
- **`simulation`** — Fake-participant parameters used only by the headless
  quick check (`run.py`); has no effect on a real session.

Every section is commented in place in the file; read the relevant comments
before changing a setting.

Each real session also writes a `*.yaml` snapshot of the settings it
actually ran with, alongside its output CSV, for reproducibility.

## 7. Where Data Is Saved

The default output folder is set by `session.output_dir` (`data`) in
`config/experiment.yaml`. Each participant gets their own subfolder, e.g.
`data/p01/`. Files produced by a session:

- `..._staircase_<timestamp>.csv` — trial-by-trial staircase history
  (stimulus level, response, correct/incorrect, reversal points).
- `..._staircase_<timestamp>.png` — staircase convergence plot.
- `..._staircase_<timestamp>.yaml` — full snapshot of the settings used in
  that session.
- `..._trialNNN_<timestamp>.csv` — per-trial raw sensor stream (force,
  finger position, timestamps) at sample resolution.

`data/` is git-ignored (only a `.gitkeep` placeholder is tracked), so
recorded data is not part of version control and stays local to disk.

## 8. Testing

```bash
pytest
```

The test suite (~16 files) covers the hardware layer most heavily
(acquisition threading, NI-DAQ AI/AO wrappers, multi-card synchronization,
voltage safety limits, Neonode communication), plus staircase logic, the
trial state machine, and data writers.

## 9. Known Gaps / To Do Before a Real Session

- **Display calibration is for the dev laptop**, not the deployment
  touchscreen — must be re-measured and updated in the `display` section
  of `config/experiment.yaml`.
