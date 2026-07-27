![BioListener Logo](data/images/BioListener_Logo_1920х200.jpg "BioListener Logo")

# BioListener Sleep: Closed-Loop Neural Stimulation System for Sleep Onset Optimization Using Deep Reinforcement Learning

A real-time closed-loop neuro-stimulation system that adapts audio-visual entrainment to the subject's live EEG, with the goal of reaching sleep onset (N1) as fast as possible. The system is built on top of the [BioListener](https://github.com/serhii-matsyshyn/biolistener) EEG board and a stroboscopic LED mask (firmware: [biolistener_sleep_mask_firmware](https://github.com/serhii-matsyshyn/biolistener_sleep_mask_firmware)) driven over the network from a Python daemon.

Pre-sleep sensory entrainment has partial precedent. Halpin et al. delivered 10 Hz stimulation before sleep as either flickering light or binaural beats, reporting a 13-minute drop in diary-recorded sleep-onset latency in an uncontrolled feasibility study ([2023](https://doi.org/10.3389/fpain.2023.1096084)), then confirmed the entrainment effect itself against a non-rhythmic sham under masked home wearable-EEG conditions ([2025](https://doi.org/10.1016/j.jpain.2025.105393)). Two caveats matter for reading them as support for this system: participants overwhelmingly chose the audio option over the visual one, and in the controlled study the sleep-onset effect was significant on diary but null on wearable EEG and actigraphy. Photic entrainment against objectively measured sleep-onset latency is therefore still largely untested. This work does not attempt to settle that question - establishing efficacy needs a powered, sham-controlled clinical trial, which is a different project. What it addresses is the control problem sitting upstream of any such trial: if a policy is going to drive stimulation from live EEG, what should it optimise, and what happens when that objective is specified naively?

The research contribution of this project is **reward design for closed-loop adaptive visual-auditory stimulation (AVS)** under a constrained-RL setting. We train Soft Actor-Critic (SAC) policies against a domain-randomised neural-mass simulator and show empirically that naive rewards from the AVS literature are exploited by the policy, while a band-weighted PLV + descent-shaped reward produces the intended frequency-descent trajectories - falling strobe frequency with a sustained rise in theta/alpha - that generalise across simulated subjects.

## Headline findings

- **Reward design matters more than the algorithm.** Three naive reward baselines (raw alpha-power, unweighted PLV, biomarker-only) all fail to descend below the alpha band (~11.5 Hz) regardless of random seed. Only the shaped reward (band-weighted PLV + descent shaping + sustained theta/alpha level) produces a policy that descends into theta.
- **Shaped reward is necessary but not sufficient.** Within the shaped variant, only 1 of 3 random seeds produces a true theta-descending policy; the other two find reward-hacking local optima. The seed with the *highest* SAC training reward (sac_eval = 290) reward-hacks at the alpha edge (final strobe 8.9 Hz); the seed with the *lowest* training reward (sac_eval = 102) is the one that descends to deep theta (final strobe 3.94 Hz, final theta/alpha ratio 5.4). The deployment policy is selected by the sleep-task metric, not the reward metric.
- **The winning policy generalises across simulated subjects.** Evaluated on 100 fresh domain-randomised NMM subjects unseen during training, the median final strobe frequency is 4.38 Hz (IQR 3.80-6.70). 68 of 100 episodes finish with the *strobe* in deep theta band (<6 Hz), 83 of 100 with it at or below the alpha-theta border (<8 Hz) - commanded stimulation frequencies, not scored brain states. Episode length is 300 s, matching real sleep-onset latency in healthy adults (5-20 min).
- **The safety supervisor is rigorous.** 10,000 adversarial stim requests in the photo-sensitive-epilepsy band [15, 25] Hz with above-cap brightness and fill - zero (0/10,000) were actuated inside the PSE band; brightness and fill were clamped to safe values on every request.
- **The brain simulator's implementation is cross-validated against the canonical reference.** Our hand-coded simulator follows the canonical Jansen & Rit (1995) equations verbatim. We cross-validated against TVB (The Virtual Brain) - the standard NMM library in computational neuroscience - and both implementations produce a bit-identical 10.74 Hz alpha peak with log-PSD correlation > 0.9. The implementation is reliable. The scope of this check is implementation fidelity against a reference implementation - the simulator's role here is as a control testbed for policy and safety-supervisor development.

## Methods

The simulator is a single-column Jansen-Rit neural mass model (pure numpy, [app/brain_sim/nmm.py](app/brain_sim/nmm.py)) augmented with state-dependent thalamocortical resonance and vigilance drift. The resonance term slides the EEG entrainment peak toward theta as drowsiness rises. Thalamocortical burst-mode physiology (Steriade et al. 1993; Bal & McCormick 1996) motivates the *firing-mode* switch, but neither paper measures a photic-driving frequency-response curve; carrying that switch over to the entrainment peak is this project's modelling hypothesis, not a fitted or borrowed result. The vigilance term has an arousal parameter modulate background drive over the session (Phillips & Robinson 2007; Hindriks and van Putten 2013). Each simulated subject samples NMM parameters, baseline arousal, and EEG noise from a domain-randomisation distribution (Tobin 2017; see Muratore et al. 2022 for the current review). The Gymnasium environment ([app/agent/env.py](app/agent/env.py)) wraps the simulator behind a 15-dimensional observation (nine band-power/biomarker features plus PLV, three last-action dims, and two temporal-context dims: EMA-smoothed PLV and delta-smoothed theta/alpha over 4 s) and a 5-dimensional continuous action (strobe frequency, duty cycle, brightness, and two audio knobs). The three photic dimensions are the ones learned against the simulator; the audio dimensions are carried through the same action vector, with master volume shaped deterministically from smoothed theta/alpha at deployment. Actions pass through a `SafetySupervisor` that enforces PSE-band exclusion, ramp-rate caps, and railed-electrode panic. We train SAC (Haarnoja 2018; stable-baselines3 implements the auto-tuned-temperature variant, Haarnoja et al. arXiv:1812.05905) via stable-baselines3 across four reward variants and three seeds each at 300-s episode length; the selected deployment policy is then evaluated on fresh held-out subjects.

This "neural model as RL environment" construction has direct precedent: Cho et al. (2024, *IEEE Trans. Neural Syst. Rehabil. Eng.* 32:3615-3624, [10.1109/TNSRE.2024.3465243](https://doi.org/10.1109/TNSRE.2024.3465243)) instantiate a basal-ganglia-thalamic model as an interactive RL environment and train SAC, TD3, PPO and A2C on it for closed-loop deep-brain stimulation. Schellenberger Costa et al. (2016, *PLoS Comput. Biol.* 12(9):e1005022, [10.1371/journal.pcbi.1005022](https://doi.org/10.1371/journal.pcbi.1005022)) likewise use a thalamocortical neural-mass model as a testbed for closed-loop sensory stimulation, validated against a human auditory-stimulation study. Luo et al. (2024, *Experiment-free exoskeleton assistance via learning in simulation*, *Nature*, [10.1038/s41586-024-07382-4](https://doi.org/10.1038/s41586-024-07382-4)) applies the same learning-in-simulation approach to a human-worn device without per-subject experiments, in a different (musculoskeletal) model class.

## Repository layout

```
app/
  agent/          SAC training, reward variants, environment, observation pipeline
  brain_sim/      Jansen-Rit NMM, photic input, domain randomisation
  eeg/            feature extraction (Welch PSD, PLV), montage, BrainFlow ingest
  audio/          pyo-based entrainment audio engine + safety-clipped controller
  mask/           UDP client for the ESP32 LED mask (and a virtual-mask server for sim)
  open_loop_track/ open-loop strobe playback for baselines (CSV / catalog driven)
  config/         pydantic schema + layered YAML loader
  persistence/    MongoDB session metadata, MNE .fif archival, parquet timeseries
  runtime/        FastAPI daemon (control loop, REST/WebSocket, lifecycle)
  cli/            biolistener-sleep entry point (run, train, eval, daemon, ...)
  ui/             React + Vite + Mantine + TanStack Query SPA, served by the daemon
configs/          YAML config files (real-subject, sim smoke, reward-ablation variants)
notebooks/        Jupyter notebooks; 03_reward_ablation.ipynb is the headline analysis
tests/            unit + integration tests; full sweep passes in ~80 s
tools/            derive_cohort.py, run_ablation.sh (12-run reward sweep)
data/external_tracks/  per-researcher symlink to an external open-loop track library
```

Runtime artefacts (gitignored): `studies/` (per-session recordings), `mlruns/` and `mlflow.db` (MLflow tracking), `app/agent/checkpoints/` (SAC `.zip` policies), `artifacts/` (eval rollouts, shield logs).

## Installation

```bash
git clone <this-repo> && cd adaptive_neuro_stimulation_sleep
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # exposes the `biolistener-sleep` CLI
pytest -q                 # full suite
```

Optional infrastructure for full-feature runs:
- **MongoDB** on `localhost:27017` for session metadata persistence (daemon will continue without it, but session creation returns 500).
- **MLflow** is filesystem-backed in `./mlruns` + `./mlflow.db` (SQLite) - no separate server required.
- **VLC** must be installed system-wide for the open-loop-track audio player (`apt install vlc` / equivalent).

## Usage

### 1. Reproduce the reward-ablation sweep

```bash
# 12 SAC training runs (4 reward variants x 3 seeds) at 5-min episodes; ~2 hr on 16 cores
bash tools/run_ablation.sh

# Pins notebooks/03_reward_ablation.ipynb to the cohort just trained
python tools/derive_cohort.py

# Cross-subject consistency on 100 fresh DR subjects; ~30 min
biolistener-sleep eval \
    --checkpoint app/agent/checkpoints/sac_full_shaped_seed3_<...>.zip \
    --n 100 --episode-s 300 \
    --out artifacts/full_shaped_rollouts.parquet

# Adversarial PSE-band attack; ~30 s
biolistener-sleep shield-test --episodes 10000 \
    --out artifacts/shield_log.json

jupyter lab notebooks/03_reward_ablation.ipynb     # Run-All produces 4 figures
```

### 2. Cross-validate the simulator against TVB

```bash
jupyter lab notebooks/01_brain_sim_validation.ipynb
```

The same comparison runs as a CI regression guard in [tests/integration/test_tvb_cross_validation.py](tests/integration/test_tvb_cross_validation.py).

### 3. Run a real-subject session

All per-subject defaults (PGA gain, audio preset + initial volume, SAC checkpoint path, real BioListener / ESP32 mask wiring) live in [configs/daemon_prod.yaml](configs/daemon_prod.yaml). The UI takes care of per-session params (subject code, protocol, EEG source, mask target, audio toggle).

```bash
biolistener-sleep daemon --config configs/daemon_prod.yaml
# Browse to http://localhost:8000/ for the React UI:
#   Subjects -> add subject -> Sessions -> create (pick protocol) -> Live (panic button stays in reach)
```

The Live page surfaces the 4 Hz control-loop tick-budget probe (last and max compute time per tick, plus an overrun counter) so the operator can immediately see if the closed loop is falling behind. The deterministic audio shaper (warmth + master volume driven by smoothed θ/α) is mirrored to telemetry too so the perceptual taper visibly responds to the subject's drowsiness.

## Validation and safety

- The `SafetySupervisor` ([app/agent/safety.py](app/agent/safety.py)) clamps every requested action before it reaches hardware: PSE band [15, 25] Hz is rerouted to the safety ceiling (14 Hz); brightness, fill, and frequency ramp rates are capped; railed electrodes trigger panic.
- The adversarial test ([app/agent/safety_adversarial.py](app/agent/safety_adversarial.py)) drives 10,000 worst-case requests through the supervisor and asserts `n_pse_leaked == 0`.
- The brain simulator is cross-validated against TVB's canonical Jansen-Rit; both produce identical alpha peaks at matched parameters (verdict: PASS, see Headline Findings).
- The test suite covers reward variants, observation-pipeline parity between training and deployment, vec-env determinism, safety supervisor behaviour, daemon-driven sessions on simulated and virtual hardware, BrainFlow wire-format compatibility, and end-to-end open-loop playback.

## Status

This is research code in active development. It has been validated in simulation and on the integration test harness; real-subject sessions are run as a feasibility appendix to the thesis, not as the primary evidence. The architecture is intentionally clear about scope: the simulation experiments are the *experimental method* (statistical power that a limited number of human sessions cannot match); the hardware testbed is the *deployment and demonstration platform*.

## License

This repository uses the following licenses:

- **Code**: [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).
- **Experimental data**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Documentation**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

This project is part of the [BioListener](https://github.com/serhii-matsyshyn/biolistener) ecosystem of open-source biosensing tooling.

For questions, contributions, or discussions, open an issue or pull request.
