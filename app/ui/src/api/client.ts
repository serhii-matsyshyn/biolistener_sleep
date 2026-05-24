// Typed REST client for the FastAPI daemon.

export interface Subject {
  _id?: string;
  code: string;
  dob_year?: number | null;
  notes?: string;
  contraindication_checklist?: Record<string, boolean>;
}

export type SessionProtocol = "baseline_no_stim" | "open_loop_track" | "rl_agent";

// audio_enabled: null = auto (daemon derives from eeg_source);
//                true = force ON; false = force OFF.
export type AudioMode = "auto" | "on" | "off";

export interface SessionDoc {
  _id?: string;
  subject_id: string;
  started_at: string;
  ended_at?: string | null;
  protocol: SessionProtocol;
  protocol_params?: Record<string, unknown>;
  eeg_source: string;
  mask_target: string;
  audio_enabled: boolean | null;
  folder_path: string;
  status: string;
}

export interface OpenLoopTrackVariant {
  key: string;
  duration_min: number | null;
  has_strobe: boolean;
}

export interface OpenLoopTrackActivity {
  name: string;
  display_name: string;
  block: string;
  variants: OpenLoopTrackVariant[];
}

// Demo Open Loop - replicates the source mask_strobe_controller Flask app
// surface (VLC audio + CSV-driven strobe) with a generative-audio toggle.

export interface DemoMaskState {
  enabled: boolean;       // "Strobe Enabled" UI toggle
  brightness: number;     // 0..1 (UI shows %)
  color: [number, number, number, number];  // R, G, B, W in 0..255
  bitmask: number;        // 8-LED mask
}

export interface DemoTrackState {
  sound_path: string | null;
  strobe_path: string | null;
  duration_s: number;
  current_pos_s: number;
  is_playing: boolean;
  first_on_s: number;
}

export interface DemoAudioState {
  generative_enabled: boolean;
  preset: string;
  master_vol_pct: number;
  binaural_vol_pct: number;
  warmth_cutoff_pct: number;
  brown_noise_pct: number;
  reverb_space_pct: number;
  test_mode_on: boolean;
}

export type DemoMaskKind = "virtual" | "esp32";

export interface DemoState {
  mask: DemoMaskState;
  track: DemoTrackState;
  audio: DemoAudioState;
  mask_kind: DemoMaskKind;
  mask_connected: boolean;
  mask_board_ip: string | null;
  safety_events: Array<{
    t_ms: number;
    kind: string;
    field: string;
    before: number;
    after: number;
    reason: string;
  }>;
}

export interface TelemetrySnapshot {
  session_id: string;
  subject_code: string;
  state: string;
  protocol?: string;
  protocol_params?: Record<string, unknown>;
  eeg_source?: string;
  mask_target?: string;
  audio_enabled?: boolean | null;
  started_at?: string;
  folder_path?: string | null;
  tick: number;
  elapsed_s: number;
  features: {
    alpha_pwr: number;
    theta_pwr: number;
    beta_pwr: number;
    hypnagogic_theta_pwr: number;
    sigma_pwr: number;
    theta_alpha_ratio: number;
    emg_rms: number;
    eog_energy: number;
    railed_pct: number;
  } | null;
  plv: number;
  restfulness: number;
  stim: {
    enabled: boolean;
    frequency_hz: number;
    fill: number;
    brightness: number;
  } | null;
  reward: number;
  // Deterministic perceptual audio shaping output (warmth + master vol).
  // Replaces the agent's master-volume action because the NMM sim has no
  // perceptual model. `drowsiness` is the saturating [0,1] signal derived
  // from smoothed θ/α that drives both knobs. Null when the session has
  // no audio shaping (e.g. demo or training).
  audio_shaper?: {
    drowsiness: number;
    warmth_pct: number;
    master_vol_pct: number;
  } | null;
  // Tick-budget probe surfaced by the daemon control loop. The closed-loop
  // ticks at 4 Hz (250 ms budget). last/max compute is the wall time spent
  // on the compute half of each tick (excludes asyncio.sleep). `overruns`
  // counts ticks where compute exceeded the budget - for a real-subject
  // session this should be 0; even a small drift means the 4 Hz tick rate
  // is degraded and PLV/feature extraction is on stale samples.
  tick_budget?: {
    last_compute_ms: number;
    max_compute_ms: number;
    overruns: number;
    budget_ms: number;
  };
  safety_events: Array<{
    t_ms: number;
    kind: string;
    field: string;
    before: number;
    after: number;
    reason: string;
  }>;
}

// All REST endpoints are namespaced under /api/ on the daemon so they
// can't collide with React Router paths (e.g. /sessions, /subjects, /live).
const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    const text = await r.text();
    // FastAPI HTTPException bodies are {"detail": "<message>"} - extract
    // just the message so consumers don't render the JSON envelope.
    let msg = text;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.detail === "string") {
        msg = parsed.detail;
      }
    } catch {
      // not JSON; keep raw text
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export const api = {
  listSubjects: () => request<Subject[]>("/subjects"),
  createSubject: (payload: Subject) =>
    request<{ id: string }>("/subjects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listSessions: (subject?: string) =>
    request<SessionDoc[]>(
      "/sessions" + (subject ? `?subject_code=${encodeURIComponent(subject)}` : ""),
    ),
  createSession: (payload: {
    subject_code: string;
    protocol?: SessionProtocol;
    protocol_params?: Record<string, unknown>;
    eeg_source?: string;
    mask_target?: string;
    audio_enabled?: boolean | null;
  }) =>
    request<{ id: string; folder: string }>("/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listOpenLoopTrackActivities: () =>
    request<OpenLoopTrackActivity[]>("/open-loop-track/activities"),

  // Demo Open Loop - replicates the source Flask app's endpoints.
  demoState: () => request<DemoState>("/demo/state"),
  demoSetMaskKind: (kind: DemoMaskKind) =>
    request<DemoState>("/demo/set_mask_kind", {
      method: "POST", body: JSON.stringify({ kind }),
    }),
  demoSelectSession: (payload: {
    sound_path: string | null;
    strobe_path: string | null;
  }) =>
    request<DemoState>("/demo/select_session", {
      method: "POST", body: JSON.stringify(payload),
    }),
  demoPlay: () =>
    request<DemoState>("/demo/play", { method: "POST" }),
  demoPause: () =>
    request<DemoState>("/demo/pause", { method: "POST" }),
  demoSeek: (pos_s: number) =>
    request<DemoState>("/demo/seek", {
      method: "POST", body: JSON.stringify({ pos_s }),
    }),
  demoUpdateParams: (payload: {
    brightness?: number; color?: number[];
    enabled?: boolean; bitmask?: number;
  }) =>
    request<DemoState>("/demo/update_params", {
      method: "POST", body: JSON.stringify(payload),
    }),
  demoUpdateAudio: (payload: Partial<DemoAudioState> & { reset?: boolean }) =>
    request<DemoState>("/demo/update_audio", {
      method: "POST", body: JSON.stringify(payload),
    }),
  demoStop: () =>
    request<DemoState>("/demo/stop", { method: "POST" }),
  listOpenLoopTrackDurations: (name: string) =>
    request<Array<{
      variant_key: string;
      duration_min: number | null;
      audio_path: string;
      strobe_path: string | null;
    }>>(`/open-loop-track/activities/${encodeURIComponent(name)}/durations`),
  startSession: (id: string) =>
    request<{ status: string }>(`/sessions/${id}/start`, { method: "POST" }),
  panicSession: (id: string) =>
    request<{ status: string }>(`/sessions/${id}/panic`, { method: "POST" }),
  panicAll: () =>
    request<{ status: string; sessions_aborted: string[] }>(
      "/panic", { method: "POST" },
    ),
  endSession: (id: string) =>
    request<{ status: string }>(`/sessions/${id}/end`, { method: "POST" }),
  sessionStatus: (id: string) =>
    request<TelemetrySnapshot>(`/sessions/${id}/status`),
};
