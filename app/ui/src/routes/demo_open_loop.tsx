/**
 * Demo Open Loop - direct mask + audio control, no session/Mongo writes.
 *
 * Replicates the source mask_strobe_controller behavior:
 *   - VLC plays the selected audio file
 *   - CsvEventPlayback drives strobe (freq, fill) from the matched CSV
 *   - 20 Hz daemon sync thread updates the mask
 *
 * Mantine-themed (consistent with the rest of the app), full-page width.
 * Generative-audio toggle replaces the track audio (VLC muted; strobe
 * keeps following the CSV on its original timeline).
 */
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Card,
  Checkbox,
  ColorInput,
  Grid,
  Group,
  Progress,
  ScrollArea,
  SegmentedControl,
  Select,
  Slider,
  Stack,
  Switch,
  Text,
  Title,
} from "@mantine/core";
import {
  IconAlertOctagon,
  IconCircleCheck,
  IconCircleX,
  IconPlayerPause,
  IconPlayerPlay,
  IconRefresh,
} from "@tabler/icons-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  api,
  type DemoMaskKind,
  type DemoState,
  type OpenLoopTrackActivity,
} from "../api/client";

interface SessionItem {
  block: string;
  name: string;
  slug: string;
  variant_key: string;
  audio_path: string;
  strobe_path: string | null;
}

function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function rgbToHex(r: number, g: number, b: number): string {
  const h = (n: number) => Math.max(0, Math.min(255, n)).toString(16).padStart(2, "0");
  return "#" + h(r) + h(g) + h(b);
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

export default function DemoOpenLoopRoute() {
  const { t } = useTranslation();
  const qc = useQueryClient();

  // Server state - 2 Hz polling drives the time/progress display.
  const stateQ = useQuery<DemoState>({
    queryKey: ["demo-state"],
    queryFn: api.demoState,
    refetchInterval: 500,
  });
  const server = stateQ.data;

  // Catalog - fetched once on mount, hydrated with per-activity variants
  // for the sessions sidebar.
  const activitiesQ = useQuery({
    queryKey: ["open-loop-track-activities"],
    queryFn: () => api.listOpenLoopTrackActivities(),
    retry: false,
  });

  const [sessions, setSessions] = useState<Record<string, SessionItem[]>>({});
  const hydrationStartedFor = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!activitiesQ.data) return;
    let cancelled = false;
    const acts: OpenLoopTrackActivity[] = activitiesQ.data;
    Promise.all(acts.map(async (act) => {
      if (hydrationStartedFor.current.has(act.name)) return [] as SessionItem[];
      hydrationStartedFor.current.add(act.name);
      try {
        const dur = await api.listOpenLoopTrackDurations(act.name);
        return dur.map<SessionItem>((v) => ({
          block: act.block,
          name: act.display_name,
          slug: act.name,
          variant_key: v.variant_key,
          audio_path: v.audio_path,
          strobe_path: v.strobe_path,
        }));
      } catch {
        return [] as SessionItem[];
      }
    })).then((batches) => {
      if (cancelled) return;
      const grouped: Record<string, SessionItem[]> = {};
      for (const batch of batches) {
        for (const item of batch) (grouped[item.block] ||= []).push(item);
      }
      setSessions(grouped);
    });
    return () => { cancelled = true; };
  }, [activitiesQ.data]);

  // Mutations
  const setStateFromResp = (data: DemoState) =>
    qc.setQueryData(["demo-state"], data);

  const selectMut = useMutation({
    mutationFn: (item: SessionItem) =>
      api.demoSelectSession({
        sound_path: item.audio_path, strobe_path: item.strobe_path,
      }),
    onSuccess: setStateFromResp,
  });
  const playMut   = useMutation({ mutationFn: api.demoPlay,   onSuccess: setStateFromResp });
  const pauseMut  = useMutation({ mutationFn: api.demoPause,  onSuccess: setStateFromResp });
  const seekMut   = useMutation({ mutationFn: api.demoSeek,   onSuccess: setStateFromResp });
  const paramMut  = useMutation({
    mutationFn: (p: Parameters<typeof api.demoUpdateParams>[0]) => api.demoUpdateParams(p),
    onSuccess: setStateFromResp,
  });
  const audioMut  = useMutation({
    mutationFn: (p: Parameters<typeof api.demoUpdateAudio>[0]) => api.demoUpdateAudio(p),
    onSuccess: setStateFromResp,
  });
  const stopMut   = useMutation({ mutationFn: api.demoStop, onSuccess: setStateFromResp });
  const maskKindMut = useMutation({
    mutationFn: (kind: DemoMaskKind) => api.demoSetMaskKind(kind),
    onSuccess: setStateFromResp,
  });

  if (!server) {
    return (
      <Stack gap="md">
        <Title order={2}>{t("demo.title")}</Title>
        <Text c="dimmed">{t("demo.loadingCatalog")}</Text>
      </Stack>
    );
  }

  const m = server.mask;
  const tr = server.track;
  const a = server.audio;

  const progressPct = tr.duration_s > 0
    ? (tr.current_pos_s / tr.duration_s) * 100
    : 0;

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-start" wrap="nowrap">
        <Stack gap={0}>
          <Title order={2}>{t("demo.title")}</Title>
          <Text size="sm" c="dimmed">{t("demo.subtitle")}</Text>
        </Stack>
        <Group gap="xs" wrap="nowrap">
          <SegmentedControl
            value={server.mask_kind}
            onChange={(v) => maskKindMut.mutate(v as DemoMaskKind)}
            data={[
              { value: "virtual", label: t("demo.maskKindVirtual") },
              { value: "esp32",   label: t("demo.maskKindEsp32") },
            ]}
            size="sm"
            disabled={maskKindMut.isPending}
          />
          <Badge
            color={server.mask_connected ? "teal" : "gray"}
            leftSection={server.mask_connected
              ? <IconCircleCheck size={14} />
              : <IconCircleX size={14} />}
            size="lg"
            variant={server.mask_connected ? "filled" : "light"}
          >
            {server.mask_connected
              ? t("demo.maskConnected", { ip: server.mask_board_ip ?? "-" })
              : t("demo.maskWaiting")}
          </Badge>
          <Badge color="orange" size="lg" variant="filled">{t("demo.notRecorded")}</Badge>
        </Group>
      </Group>

      {server.mask_kind === "esp32" && !server.mask_connected && (
        <Alert color="yellow" variant="light"
               title={t("demo.maskAlertTitle")}>
          {t("demo.maskAlertBody")}
        </Alert>
      )}

      <Grid>
        {/* ─────────────── Sessions ─────────────── */}
        <Grid.Col span={{ base: 12, md: 4, lg: 3 }}>
          <Card withBorder p="md">
            <Title order={4} mb="sm">{t("demo.sessions")}</Title>
            <ScrollArea h={620} type="auto" scrollbarSize={6}>
              <Stack gap={2}>
                {Object.entries(sessions).map(([block, items]) => (
                  <Stack key={block} gap={2} mb="md">
                    <Text size="xs" fw={700} c="teal.4"
                          tt="uppercase" mt="xs">
                      {block}
                    </Text>
                    {items.map((item) => {
                      const isActive = item.audio_path === tr.sound_path;
                      return (
                        <Card key={`${item.slug}/${item.variant_key}`}
                              p="xs" withBorder={isActive}
                              style={{
                                cursor: "pointer",
                                background: isActive
                                  ? "rgba(20, 184, 166, 0.08)"
                                  : undefined,
                              }}
                              onClick={() => selectMut.mutate(item)}>
                          <Text size="sm">{item.name}</Text>
                          <Text size="xs" c="dimmed">{item.variant_key}</Text>
                        </Card>
                      );
                    })}
                  </Stack>
                ))}
                {activitiesQ.isLoading && (
                  <Text size="sm" c="dimmed">{t("demo.loadingCatalog")}</Text>
                )}
                {activitiesQ.error && (
                  <Alert color="red" variant="light" mt="sm">
                    {t("demo.catalogUnavailable", { err: String(activitiesQ.error) })}
                  </Alert>
                )}
              </Stack>
            </ScrollArea>
          </Card>
        </Grid.Col>

        {/* ─────────────── Playback ─────────────── */}
        <Grid.Col span={{ base: 12, md: 8, lg: 9 }}>
          <Stack gap="md">
            {/* Playback transport */}
            <Card withBorder p="md">
              <Stack gap="md">
                <Title order={4}>{t("demo.playback")}</Title>
                <Group>
                  <Button
                    leftSection={<IconPlayerPlay size={18} />}
                    disabled={!tr.sound_path || tr.is_playing}
                    onClick={() => playMut.mutate()}
                  >
                    {t("demo.play")}
                  </Button>
                  <Button
                    leftSection={<IconPlayerPause size={18} />}
                    variant="default"
                    disabled={!tr.is_playing}
                    onClick={() => pauseMut.mutate()}
                  >
                    {t("demo.pause")}
                  </Button>
                  <Text size="sm" c="dimmed" ff="monospace">
                    {fmtTime(tr.current_pos_s)} / {fmtTime(tr.duration_s)}
                  </Text>
                  {tr.sound_path && (
                    <Text size="xs" c="dimmed" truncate style={{ flex: 1 }}>
                      {tr.sound_path.split("/").pop()}
                    </Text>
                  )}
                </Group>
                <Progress
                  value={progressPct}
                  size="sm"
                  radius="xl"
                  style={{ cursor: tr.duration_s > 0 ? "pointer" : "default" }}
                  onClick={(e) => {
                    if (tr.duration_s <= 0) return;
                    const r = e.currentTarget.getBoundingClientRect();
                    const pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
                    seekMut.mutate(pct * tr.duration_s);
                  }}
                />
              </Stack>
            </Card>

            {/* Strobe controls */}
            <Card withBorder p="md">
              <Stack gap="md">
                <Group justify="space-between">
                  <Title order={4}>{t("demo.strobe")}</Title>
                  <Switch
                    checked={m.enabled}
                    onChange={(e) =>
                      paramMut.mutate({ enabled: e.currentTarget.checked })}
                    label={t("demo.enabled")}
                  />
                </Group>

                <div>
                  <Text size="sm" mb={2}>
                    {t("demo.brightness")}: {Math.round(m.brightness * 100)}%
                  </Text>
                  <Slider
                    value={Math.round(m.brightness * 100)}
                    min={0} max={100}
                    onChange={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                      s ? { ...s, mask: { ...s.mask, brightness: v / 100 } } : s)}
                    onChangeEnd={(v) => paramMut.mutate({ brightness: v / 100 })}
                  />
                </div>

                <Group grow align="end">
                  <ColorInput
                    label={t("demo.rgbColor")}
                    format="hex"
                    value={rgbToHex(m.color[0], m.color[1], m.color[2])}
                    onChange={(hex) => {
                      const [r, g, b] = hexToRgb(hex);
                      paramMut.mutate({ color: [r, g, b, m.color[3]] });
                    }}
                  />
                  <div>
                    <Text size="sm" mb={2}>{t("demo.whiteChannel")}: {m.color[3]}</Text>
                    <Slider
                      value={m.color[3]}
                      min={0} max={255}
                      onChange={(w) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, mask: {
                          ...s.mask,
                          color: [s.mask.color[0], s.mask.color[1], s.mask.color[2], w],
                        } } : s)}
                      onChangeEnd={(w) =>
                        paramMut.mutate({ color: [m.color[0], m.color[1], m.color[2], w] })}
                    />
                  </div>
                </Group>

                <div>
                  <Text size="sm" mb={6}>{t("demo.ledBitmask")}</Text>
                  <Group gap="md">
                    {Array.from({ length: 8 }, (_, i) => {
                      const bit = 1 << i;
                      const on = (m.bitmask & bit) !== 0;
                      return (
                        <Checkbox
                          key={i}
                          label={`LED ${i}`}
                          checked={on}
                          onChange={(e) => {
                            const next = e.currentTarget.checked
                              ? (m.bitmask | bit) : (m.bitmask & ~bit);
                            paramMut.mutate({ bitmask: next });
                          }}
                        />
                      );
                    })}
                  </Group>
                </div>
              </Stack>
            </Card>

            {/* Generative audio (replaces the track) */}
            <Card withBorder p="md">
              <Stack gap="md">
                <Group justify="space-between">
                  <div>
                    <Title order={4}>{t("demo.generativeTitle")}</Title>
                    <Text size="xs" c="dimmed">{t("demo.generativeHelp")}</Text>
                  </div>
                  <Switch
                    checked={a.generative_enabled}
                    onChange={(e) =>
                      audioMut.mutate({ generative_enabled: e.currentTarget.checked })}
                    label={t("demo.enable")}
                  />
                </Group>

                {a.generative_enabled && (
                  <Stack gap="md">
                    <Group>
                      <Select
                        label={t("demo.preset")}
                        data={[
                          { value: "sunlight_pads", label: "Sunlight Pads (C maj 9)" },
                          { value: "warm_blanket",  label: "Warm Blanket (G sus 2)" },
                          { value: "lunar_bowls",   label: "Lunar Bowls (F maj pentatonic)" },
                        ]}
                        value={a.preset}
                        onChange={(v) => v && audioMut.mutate({ preset: v })}
                        style={{ flex: 1 }}
                      />
                      <SegmentedControl
                        value={a.test_mode_on ? "on" : "off"}
                        onChange={(v) => audioMut.mutate({ test_mode_on: v === "on" })}
                        data={[
                          { value: "off", label: t("demo.pingPongOff") },
                          { value: "on",  label: t("demo.pingPongOn") },
                        ]}
                        mt={22}
                      />
                      <ActionIcon
                        variant="default" mt={22} size="lg"
                        onClick={() => audioMut.mutate({ reset: true })}
                        title={t("demo.resetDefaults")}
                      >
                        <IconRefresh size={18} />
                      </ActionIcon>
                    </Group>

                    <GenSlider label={t("demo.masterVol")} value={a.master_vol_pct}
                      field="master_vol_pct"
                      onDrag={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, audio: { ...s.audio, master_vol_pct: v } } : s)}
                      onCommit={(v) => audioMut.mutate({ master_vol_pct: v })} />
                    <GenSlider label={t("demo.binauralVol")} value={a.binaural_vol_pct}
                      field="binaural_vol_pct"
                      onDrag={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, audio: { ...s.audio, binaural_vol_pct: v } } : s)}
                      onCommit={(v) => audioMut.mutate({ binaural_vol_pct: v })} />
                    <GenSlider label={t("demo.warmth")} value={a.warmth_cutoff_pct}
                      field="warmth_cutoff_pct"
                      onDrag={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, audio: { ...s.audio, warmth_cutoff_pct: v } } : s)}
                      onCommit={(v) => audioMut.mutate({ warmth_cutoff_pct: v })} />
                    <GenSlider label={t("demo.brownNoise")} value={a.brown_noise_pct}
                      field="brown_noise_pct"
                      onDrag={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, audio: { ...s.audio, brown_noise_pct: v } } : s)}
                      onCommit={(v) => audioMut.mutate({ brown_noise_pct: v })} />
                    <GenSlider label={t("demo.reverb")} value={a.reverb_space_pct}
                      field="reverb_space_pct"
                      onDrag={(v) => qc.setQueryData<DemoState>(["demo-state"], (s) =>
                        s ? { ...s, audio: { ...s.audio, reverb_space_pct: v } } : s)}
                      onCommit={(v) => audioMut.mutate({ reverb_space_pct: v })} />
                  </Stack>
                )}
              </Stack>
            </Card>

            <Button
              color="red" size="lg" fullWidth
              leftSection={<IconAlertOctagon size={20} />}
              loading={stopMut.isPending}
              onClick={() => stopMut.mutate()}
            >
              {t("demo.stopAll")}
            </Button>
          </Stack>
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

function GenSlider(props: {
  label: string;
  value: number;
  field: keyof DemoState["audio"];
  onDrag: (v: number) => void;
  onCommit: (v: number) => void;
}) {
  // Suppress field-unused TS warning while keeping it as a discriminator
  // for future per-field side-effects (e.g. logging).
  void props.field;
  return (
    <div>
      <Text size="sm" mb={2}>
        {props.label}: {Math.round(props.value)}%
      </Text>
      <Slider
        value={Math.round(props.value)}
        min={0} max={100}
        onChange={props.onDrag}
        onChangeEnd={props.onCommit}
      />
    </div>
  );
}

