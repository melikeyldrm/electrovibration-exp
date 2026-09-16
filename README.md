# electrovibration-exp

Psychophysics rig for measuring electrovibration detection thresholds on a
touchscreen, using a 2-alternative-forced-choice (2AFC) staircase procedure.
A participant slides a finger through two intervals (one carries the
electrovibration stimulus, one does not) and reports which one felt
different. An adaptive staircase adjusts the stimulus voltage trial by trial
to converge on the participant's detection threshold.

The rig also records the participant's finger position and the normal force
they press with (via two ATI Nano17 force sensors), so trials can be
reviewed for grip force, timing, and touch trajectory alongside the
psychophysical response.

Experiment `exp01_2afc_threshold` implements the specific protocol currently
being run. Later experiments would live alongside it under `experiments/`.

## Project layout

```
config/experiment.yaml        Single YAML config: session, display calibration,
                               staircase, trial timing, force feedback, acquisition.
experiments/
  exp01_2afc_threshold/
    run.py                    Headless simulated run (no GUI, no hardware) - checks
                               that the staircase/trial/logging pipeline works end to end.
    run_gui.py                Full interactive session: real or mocked DAQ hardware,
                               experimenter + participant windows, live data logging.
src/evexp/
  hardware/                   Device layer: DAQ cards (NI-DAQmx), force sensors,
                               finger-position tracking (mouse or Neonode IR sensor),
                               screen calibration, voltage safety limits, mock devices
                               for developing without hardware attached.
  psychophysics/               Trial state machine (Trial2AFC) and the adaptive
                               staircase controller.
  processing/                  Signal processing: force-band feedback colouring,
                               waveform helpers.
  data/                        Trial loggers: per-session CSV summary and
                               raw per-trial CSV writer for the full sensor stream.
  ui/                          PyQt5 windows: participant-facing display, experimenter
                               console, session orchestration (SessionController).
scripts/                      One-off hardware probing/debugging scripts (Neonode HID
                               probing, DAQ smoke tests, staircase tracing). Not part of
                               the experiment pipeline; useful when bringing up new hardware.
tests/                        pytest suite - primarily covers the hardware layer,
                               staircase logic, and data writers.
data/                         Recorded session output (git-ignored; only a
                               .gitkeep placeholder is tracked).
```

## Setup

Requires Python 3.9+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -e .
```

`nidaqmx` and `hidapi` are only needed for real hardware (`--real-daq` /
`--neonode`); everything else runs against mocked devices without them.

## Running

**Headless smoke test** — runs a simulated observer through the full
staircase with no GUI and no hardware attached. Use this to verify the
trial/staircase/logging pipeline after a code change, before touching a
real session:

```bash
python experiments/exp01_2afc_threshold/run.py
```

Writes a CSV and a convergence plot to `data/`.

**Interactive session** — the real experiment runner, with a participant
window and an experimenter console:

```bash
python experiments/exp01_2afc_threshold/run_gui.py
```

By default this uses mocked DAQ hardware and the mouse for finger position,
so it runs on a laptop with nothing plugged in. Useful flags:

| Flag | Effect |
|---|---|
| `--real-daq` | Use real (or NI MAX-simulated) NI-DAQmx cards instead of the mock devices, for both acquisition and the stimulus. |
| `--daq-fs1`, `--daq-fs2`, `--daq-stim` | NI-DAQmx device names for the two force-sensor cards and the stimulus/monitor card (defaults: `Dev1`, `Dev3`, `Dev2`). Only relevant with `--real-daq`. |
| `--monitor-channel` | AI channel read as `current` (default `ai0`). |
| `--neonode` | Track finger position with the Neonode IR sensor instead of the mouse; falls back to the mouse if the sensor can't be opened. |


| `--list-screens` | Print index/geometry of every connected monitor, then exit — use this to find `display.participant_screen_index` before a real session, then Ctrl+C or close the window. |

Ctrl+C and closing the participant window both trigger a clean shutdown
(DAQ tasks closed, stats printed, in-progress trial not silently dropped).

## Configuration

Everything session-specific lives in [`config/experiment.yaml`](config/experiment.yaml):
display calibration (physical screen size in mm — critical, since cue speed
and track length are computed from it), the staircase rule and step sizes,
trial timing, force-feedback target/band, and acquisition sample rate. The
file is heavily commented in place; read it before changing anything,
especially the `display` section, which is currently calibrated for a dev
laptop and **must be re-measured** against the real touchscreen before any
real session (see the `TODO` comment in that section).

Each real session also writes a `*.yaml` snapshot of the config it actually
ran with, alongside its output CSV, for reproducibility.

## Testing

```bash
pytest
```

The suite (~16 test files) covers the hardware layer most heavily —
acquisition threading, the NI-DAQ AI/AO wrappers, multi-card synchronization,
voltage safety limits, the Neonode transport — plus the staircase, trial
state machine, and data writers.

## Known gaps / things to fix before a real session

- **Display calibration is for the dev laptop**, not the deployment
  touchscreen — see `config/experiment.yaml`'s `display` section.

Force calibration matrices (`GAIN_FS1`, `GAIN_FS2` in
[`src/evexp/hardware/force.py`](src/evexp/hardware/force.py)) are the real
per-sensor gains; update them there if a sensor is replaced or re-calibrated.
