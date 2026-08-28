/**
 * Hardware-session protocols.
 *
 * Covers the parts of a session a terminal is bad at: checking electrode
 * quality before lying down, starting without retyping a command, watching the
 * block timeline, and reading the gate verdict afterwards.
 *
 * The subject cannot see this page during the experimental session - they are
 * motionless with eyes closed under the mask, guided by audio. The screen is
 * for setup, launch, and review, and for the validation session, which allows
 * movement.
 */
import {
  Alert,
  Badge,
  Button,
  Card,
  Switch,
  Code,
  Group,
  NumberInput,
  Progress,
  Input,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { TelemetryTimeseries } from "../components/TelemetryTimeseries";
import type { HistorySample } from "../hooks/useTelemetryHistory";

type Block = {
  name: string;
  kind: string;
  duration_s: number;
  eyes: string;
  kss_at_end: boolean;
  note: string;
  t_start_s: number;
  t_end_s: number;
};

const FIELD_STYLES = {
  root: { height: "100%", display: "flex", flexDirection: "column" },
  wrapper: { marginTop: "auto", paddingTop: 6, marginBottom: 0 },
} as const;

type Plan = {
  ok: boolean;
  suggested_order: string;
  default_checkpoint: string;
  order_history: Record<string, number>;
  problems: string[];
  warnings: string[];
  missing_cues: string[];
  protocol: string;
  total_duration_s: number;
  blocks: Block[];
};

type Tick = Record<string, number | string | null>;

type Status = {
  running: boolean;
  status: string;
  protocol: string | null;
  session_dir: string | null;
  elapsed_s: number | null;
  blocks: Block[];
  current: {
    block: string;
    kind: string;
    elapsed_s: number;
    duration_s: number;
    phase: string;
    probe_hz: number | null;
  } | null;
  tick: Tick | null;
  kss: { t_s: number; block: string; value: number }[];
  kss_pending: string | null;
  events: { t: number; kind: string; [k: string]: unknown }[];
  error: string | null;
  abort_reason: string | null;
};

const API = "/api/protocol";

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

async function postJSON<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

/** Protocol ticks carry more than the plots need; map to what they expect. */
function toSample(tick: Tick): HistorySample {
  const num = (k: string) => Number(tick[k] ?? 0) || 0;
  return {
    t_s: num("t_s"),
    alpha: num("alpha_pwr"),
    theta: num("theta_pwr"),
    beta: num("beta_pwr"),
    hypno_theta: num("hypnagogic_theta_pwr"),
    sigma: num("sigma_pwr"),
    theta_alpha_ratio: num("theta_alpha_ratio"),
    plv: num("plv"),
    restfulness: 0,
    freq_hz: num("act_freq_hz"),
    brightness: num("act_brightness"),
    fill: num("act_fill"),
    reward: num("reward"),
  };
}

function fmtMin(s: number | null | undefined) {
  if (s == null) return "-";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

export default function ProtocolRoute() {
  const { t } = useTranslation();
  const qc = useQueryClient();

  const [protocol, setProtocol] = useState("validation");
  const [order, setOrder] = useState("auto");
  const [eeg, setEeg] = useState("sim");
  const [mask, setMask] = useState("virtual");
  const [checkpoint, setCheckpoint] = useState("");
  const [ckTouched, setCkTouched] = useState(false);
  const [amplitudeScale, setAmplitudeScale] = useState<string>("");
  const [seed, setSeed] = useState<number>(0);
  const [compressFactor, setCompressFactor] = useState<number>(1);
  const [audioCues, setAudioCues] = useState(true);

  const planQ = useQuery<Plan>({
    queryKey: ["protocol-plan", protocol, order, seed, compressFactor, checkpoint, eeg, audioCues, amplitudeScale],
    queryFn: () =>
      getJSON<Plan>(
        `${API}/plan?` +
          new URLSearchParams({
            protocol,
            order,
            seed: String(seed),
            compress_factor: String(compressFactor),
            eeg,
            audio_cues: String(audioCues),
            ...(checkpoint ? { checkpoint } : {}),
            ...(amplitudeScale ? { amplitude_scale: amplitudeScale } : {}),
          }),
      ),
  });

  // Poll rather than subscribe: the payload is small and a dropped frame costs
  // nothing, where a stale socket would silently freeze the timeline.
  const statusQ = useQuery<Status>({
    queryKey: ["protocol-status"],
    queryFn: () => getJSON<Status>(`${API}/status`),
    refetchInterval: 500,
  });
  const st = statusQ.data;

  useEffect(() => {
    const advertised = planQ.data?.default_checkpoint;
    if (advertised && !ckTouched && !checkpoint) setCheckpoint(advertised);
  }, [planQ.data?.default_checkpoint, ckTouched, checkpoint]);

  const start = useMutation({
    mutationFn: () =>
      postJSON(`${API}/start`, {
        protocol,
        order,
        seed,
        eeg,
        mask,
        amplitude_scale: amplitudeScale ? Number(amplitudeScale) : null,
        checkpoint: checkpoint || null,
        audio_cues: audioCues,
        compress_factor: compressFactor,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["protocol-status"] }),
  });
  const stop = useMutation({
    mutationFn: () => postJSON(`${API}/stop`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["protocol-status"] }),
  });
  const kss = useMutation({
    mutationFn: (value: number) => postJSON(`${API}/kss`, { value }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["protocol-status"] }),
  });

  // Accumulate ticks locally; the server keeps a bounded window of its own.
  const [history, setHistory] = useState<HistorySample[]>([]);
  const lastT = useRef<number>(-1);
  useEffect(() => {
    const tick = st?.tick;
    if (!tick) return;
    const ts = Number(tick.t_s ?? 0);
    if (ts <= lastT.current) return;
    lastT.current = ts;
    setHistory((h) => [...h.slice(-1200), toSample(tick)]);
  }, [st?.tick]);
  useEffect(() => {
    if (!st?.running) return;
    setHistory([]);
    lastT.current = -1;
  }, [st?.session_dir, st?.protocol]);

  const blocks = st?.running && st.blocks.length ? st.blocks : planQ.data?.blocks ?? [];
  const current = st?.current;
  const railed = Number(st?.tick?.railed_pct ?? 0);

  const sessionPct = useMemo(() => {
    if (!st?.tick) return 0;
    const e = Number(st.tick.session_elapsed_s ?? 0);
    const d = Number(st.tick.session_duration_s ?? 1);
    return d > 0 ? Math.min(100, (e / d) * 100) : 0;
  }, [st?.tick]);

  const running = !!st?.running;

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("protocol.title")}</Title>
        <Group>
          {running ? (
            <Badge color="green" size="lg">
              {t("protocol.running")} · {fmtMin(st?.tick ? Number(st.tick.session_elapsed_s) : null)}
            </Badge>
          ) : (
            <Badge color="gray" size="lg">
              {st?.status ?? "idle"}
            </Badge>
          )}
          {running ? (
            <Button color="red" onClick={() => stop.mutate()}>
              {t("protocol.stop")}
            </Button>
          ) : (
            <Button
              disabled={!planQ.data?.ok || start.isPending}
              onClick={() => start.mutate()}
            >
              {t("protocol.start")}
            </Button>
          )}
        </Group>
      </Group>

      {st?.error && (
        <Alert color="red" title={t("protocol.failed")}>
          <Code>{st.error}</Code>
        </Alert>
      )}
      {st?.abort_reason && !running && (
        <Alert color="yellow" title={t("protocol.aborted")}>
          {st.abort_reason}
        </Alert>
      )}

      {!running && (
        <Card withBorder>
          <Title order={4} mb="sm">
            {t("protocol.setup")}
          </Title>
          {/* Every cell carries a label and a description of the same height,
              so the inputs share one baseline across the row. */}
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md" verticalSpacing="md">
            <Select
              label={t("protocol.which")}
              description={t("protocol.whichHint")}
              styles={FIELD_STYLES}
              value={protocol}
              onChange={(v) => setProtocol(v ?? "validation")}
              data={[
                { value: "validation", label: t("protocol.validation") },
                { value: "full", label: t("protocol.full") },
              ]}
            />
            <Select
              label={t("protocol.armOrder")}
              description={t("protocol.armOrderHint")}
              styles={FIELD_STYLES}
              value={order}
              disabled={protocol !== "full"}
              onChange={(v) => setOrder(v ?? "ramp_first")}
              data={[
                { value: "auto", label: t("protocol.orderAuto") },
                { value: "abba", label: "ABBA (ramp outer)" },
                { value: "baab", label: "BAAB (agent outer)" },
              ]}
            />
            <Select
              label={t("protocol.eegSource")}
              description={t("protocol.eegSourceHint")}
              styles={FIELD_STYLES}
              value={eeg}
              onChange={(v) => setEeg(v ?? "sim")}
              data={["sim", "biolistener", "playback"]}
            />
            <Input.Wrapper
              label={t("protocol.mask")}
              description={t("protocol.maskHint")}
              styles={FIELD_STYLES}
            >
              <SegmentedControl
                mt="auto"
                fullWidth
                value={mask}
                onChange={setMask}
                data={[
                  { value: "virtual", label: t("demo.maskKindVirtual") },
                  { value: "esp32", label: t("demo.maskKindEsp32") },
                ]}
              />
            </Input.Wrapper>
            <TextInput
              label={t("protocol.amplitudeScale")}
              description={t("protocol.amplitudeScaleHint")}
              styles={FIELD_STYLES}
              value={amplitudeScale}
              onChange={(e) => setAmplitudeScale(e.currentTarget.value)}
              placeholder="17.7"
              error={
                eeg === "biolistener" && protocol === "full" && !amplitudeScale
                  ? t("protocol.amplitudeScaleMissing")
                  : undefined
              }
            />
            <TextInput
              label={t("protocol.checkpoint")}
              description={t("protocol.checkpointHint")}
              styles={FIELD_STYLES}
              value={checkpoint}
              onChange={(e) => { setCkTouched(true); setCheckpoint(e.currentTarget.value); }}
              placeholder="app/agent/checkpoints/...zip"
              disabled={protocol !== "full"}
            />
            <NumberInput
              label={t("protocol.seed")}
              description={t("protocol.seedHint")}
              styles={FIELD_STYLES}
              value={seed}
              onChange={(v) => setSeed(Number(v) || 0)}
              min={0}
            />
            <NumberInput
              label={t("protocol.compress")}
              description={t("protocol.compressHint")}
              styles={FIELD_STYLES}
              value={compressFactor}
              onChange={(v) => setCompressFactor(Number(v) || 1)}
              min={0.01}
              max={1}
              step={0.05}
              decimalScale={2}
            />
            <Input.Wrapper
              label={t("protocol.audioCues")}
              description={t("protocol.audioCuesHint")}
              styles={FIELD_STYLES}
            >
              <Switch
                mt="auto"
                checked={audioCues}
                onChange={(e) => setAudioCues(e.currentTarget.checked)}
                onLabel="ON"
                offLabel="OFF"
                size="lg"
              />
            </Input.Wrapper>
          </SimpleGrid>

          {planQ.data && (
            <Stack gap="xs" mt="md">
              <Text size="sm">
                {planQ.data.protocol} · {blocks.length} {t("protocol.blocks")} ·{" "}
                {(planQ.data.total_duration_s / 60).toFixed(1)} {t("protocol.min")}
              </Text>
              {protocol === "full" && planQ.data.order_history && (
                <Text size="xs" c="dimmed">
                  {t("protocol.nextOrder", { order: planQ.data.suggested_order })} ·{" "}
                  {Object.entries(planQ.data.order_history)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(", ")}
                </Text>
              )}
              {mask === "esp32" && (
                <Alert color="yellow" variant="light"
                       title={t("demo.maskAlertTitle")}>
                  {t("demo.maskAlertBody")}
                </Alert>
              )}
              {planQ.data.problems.map((p) => (
                <Alert key={p} color="red" variant="light" py={6}>
                  {p}
                </Alert>
              ))}
              {planQ.data.warnings.map((w) => (
                <Alert key={w} color="yellow" variant="light" py={6}>
                  {w}
                </Alert>
              ))}
              {planQ.data.ok && planQ.data.problems.length === 0 && (
                <Alert color="green" variant="light" py={6}>
                  {t("protocol.preflightOk")}
                </Alert>
              )}
            </Stack>
          )}
        </Card>
      )}

      {running && (
        <>
          <Card withBorder>
            <Group justify="space-between" mb={4}>
              <Text fw={600}>
                {current?.block ?? "-"}{" "}
                <Text span c="dimmed" size="sm">
                  {current?.kind}
                  {current?.phase ? ` · ${current.phase}` : ""}
                  {current?.probe_hz != null ? ` · ${current.probe_hz} Hz` : ""}
                </Text>
              </Text>
              <Text size="sm" c="dimmed">
                {fmtMin(current?.elapsed_s)} / {fmtMin(current?.duration_s)} ·{" "}
                {t("protocol.session")} {fmtMin(Number(st?.tick?.session_elapsed_s ?? 0))} /{" "}
                {fmtMin(Number(st?.tick?.session_duration_s ?? 0))}
              </Text>
            </Group>
            <Progress
              value={current ? (current.elapsed_s / Math.max(current.duration_s, 1)) * 100 : 0}
              mb={6}
            />
            <Progress value={sessionPct} color="gray" size="sm" />
          </Card>

          <SimpleGrid cols={{ base: 2, sm: 6 }}>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                {t("protocol.railed")}
              </Text>
              <Text fw={700} c={railed >= 20 ? "red" : undefined}>
                {railed.toFixed(0)} %
              </Text>
            </Card>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                {t("protocol.thetaAlpha")}
              </Text>
              <Text fw={700}>{Number(st?.tick?.theta_alpha_ratio ?? 0).toFixed(3)}</Text>
            </Card>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                PLV
              </Text>
              <Text fw={700}>{Number(st?.tick?.plv ?? 0).toFixed(3)}</Text>
            </Card>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                {t("protocol.reward")}
              </Text>
              <Text fw={700}>{Number(st?.tick?.reward ?? 0).toFixed(2)}</Text>
            </Card>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                {t("protocol.dose")}
              </Text>
              <Text fw={700}>{Number(st?.tick?.dose ?? 0).toFixed(3)}</Text>
            </Card>
            <Card withBorder padding="xs">
              <Text size="xs" c="dimmed">
                {t("protocol.clamps")}
              </Text>
              <Text fw={700}>{Number(st?.tick?.n_safety_events ?? 0)}</Text>
            </Card>
          </SimpleGrid>

          {st?.kss_pending && (
            <Alert color="blue" title={t("protocol.kssPrompt", { block: st.kss_pending })}>
              <Group gap="xs" mt="xs">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((v) => (
                  <Button key={v} size="xs" variant="light" onClick={() => kss.mutate(v)}>
                    {v}
                  </Button>
                ))}
              </Group>
            </Alert>
          )}

          <Card withBorder>
            <TelemetryTimeseries history={history} />
          </Card>
        </>
      )}

      <Card withBorder>
        <Title order={5} mb="xs">
          {t("protocol.timeline")}
        </Title>
        <Table striped withTableBorder fz="xs">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{t("protocol.block")}</Table.Th>
              <Table.Th>{t("protocol.kind")}</Table.Th>
              <Table.Th>{t("protocol.from")}</Table.Th>
              <Table.Th>{t("protocol.length")}</Table.Th>
              <Table.Th>{t("protocol.eyes")}</Table.Th>
              <Table.Th>KSS</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {blocks.map((b) => {
              const active = running && current?.block === b.name;
              return (
                <Table.Tr key={b.name} bg={active ? "var(--mantine-color-blue-light)" : undefined}>
                  <Table.Td fw={active ? 700 : undefined}>{b.name}</Table.Td>
                  <Table.Td>{b.kind}</Table.Td>
                  <Table.Td>{fmtMin(b.t_start_s)}</Table.Td>
                  <Table.Td>{fmtMin(b.duration_s)}</Table.Td>
                  <Table.Td>{b.eyes}</Table.Td>
                  <Table.Td>{b.kss_at_end ? "•" : ""}</Table.Td>
                </Table.Tr>
              );
            })}
          </Table.Tbody>
        </Table>
      </Card>

      {st?.session_dir && (
        <Card withBorder>
          <Text size="sm">
            {t("protocol.recordedTo")} <Code>{st.session_dir}</Code>
          </Text>
          {st.kss.length > 0 && (
            <Text size="sm" mt={4}>
              {t("protocol.ratings")}: {st.kss.map((k) => `${k.block}=${k.value}`).join(", ")}
            </Text>
          )}
        </Card>
      )}
    </Stack>
  );
}
