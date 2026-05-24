import {
  Alert,
  Badge,
  Card,
  Code,
  Grid,
  Group,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useState, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { BandPowerBars } from "../components/BandPowerBars";
import { PanicButton } from "../components/PanicButton";
import { PlvGauge } from "../components/PlvGauge";
import { TelemetryTimeseries } from "../components/TelemetryTimeseries";
import { useLiveTelemetry } from "../hooks/useLiveTelemetry";
import { useTelemetryHistory } from "../hooks/useTelemetryHistory";
import type { TelemetrySnapshot } from "../api/client";

export default function LiveRoute() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  // URL is the source of truth: /live/:sessionId. The TextInput in the
  // header is a "switch session" affordance - pressing Enter pushes a new URL.
  const { sessionId: urlSessionId } = useParams();
  const sessionId = urlSessionId ?? null;
  const [switchTo, setSwitchTo] = useState<string>("");
  const telemetry = useLiveTelemetry(sessionId);
  const snap = telemetry.data ?? null;
  const history = useTelemetryHistory(sessionId);

  const onSwitch = () => {
    const trimmed = switchTo.trim();
    if (trimmed) navigate(`/live/${trimmed}`);
  };

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("app.nav.live")}</Title>
        <TextInput
          placeholder={sessionId ?? t("live.switchPlaceholder")}
          value={switchTo}
          onChange={(e) => setSwitchTo(e.currentTarget.value.trim())}
          onKeyDown={(e) => { if (e.key === "Enter") onSwitch(); }}
          w={300}
        />
      </Group>

      {!sessionId && (
        <Alert color="yellow" title={t("live.noSession")}>
          {t("live.noSessionHelp")}
        </Alert>
      )}
      {sessionId && !snap && (
        <Alert color="gray" title={t("live.connecting")}>
          {t("live.connectingHelp")}
        </Alert>
      )}

      {snap && (
        <Grid>
          <Grid.Col span={12}>
            <SessionDetailsCard snap={snap} />
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 8 }}>
            <Stack gap="sm">
              <Card withBorder>
                <BandPowerBars snap={snap} />
              </Card>
              <Card withBorder>
                <PlvGauge plv={snap.plv} />
              </Card>
            </Stack>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 4 }}>
            <Stack gap="sm">
              <Card withBorder>
                <Stack gap="xs">
                  <Group justify="space-between">
                    <Text fw={500}>{t("live.state")}</Text>
                    <Badge
                      color={
                        snap.state === "running"
                          ? "teal"
                          : snap.state === "aborted"
                            ? "red"
                            : "gray"
                      }
                    >
                      {snap.state}
                    </Badge>
                  </Group>
                  <Text size="sm" c="dimmed">
                    {t("live.tick")}: {snap.tick} | {t("live.elapsed")}:{" "}
                    {snap.elapsed_s.toFixed(1)}s
                  </Text>
                  {snap.features && (
                    <>
                      <Text size="sm">
                        {t("live.thetaAlphaRatio")}:{" "}
                        {snap.features.theta_alpha_ratio.toFixed(3)}
                      </Text>
                      <Text size="sm">{t("live.plv")}: {snap.plv.toFixed(3)}</Text>
                      <Text size="sm">
                        {t("live.restfulness")}: {snap.restfulness.toFixed(3)}
                      </Text>
                    </>
                  )}
                  {snap.features && (
                    <Stack gap={4}>
                      <Text size="xs" c="dimmed">
                        railed {snap.features.railed_pct.toFixed(0)}%
                      </Text>
                      <Progress
                        value={snap.features.railed_pct}
                        color={snap.features.railed_pct > 50 ? "red" : "teal"}
                        size="md"
                      />
                    </Stack>
                  )}
                </Stack>
              </Card>

              {snap.stim && (
                <Card withBorder>
                  <Stack gap="xs">
                    <Text fw={500}>{t("live.stim")}</Text>
                    <Text size="sm">
                      {t("live.frequency")}: {snap.stim.frequency_hz.toFixed(2)}
                    </Text>
                    <Text size="sm">
                      {t("live.fill")}: {(snap.stim.fill * 100).toFixed(1)}%
                    </Text>
                    <Text size="sm">
                      {t("live.brightness")}: {(snap.stim.brightness * 100).toFixed(1)}%
                    </Text>
                    <Text size="sm">
                      {t("live.reward")}: {snap.reward.toFixed(4)}
                    </Text>
                  </Stack>
                </Card>
              )}

              {snap.audio_shaper && (
                <Card withBorder>
                  <Stack gap="xs">
                    <Text fw={500}>{t("live.audioShaper")}</Text>
                    <Text size="sm">
                      {t("live.drowsiness")}: {(snap.audio_shaper.drowsiness * 100).toFixed(0)}%
                    </Text>
                    <Text size="sm">
                      {t("live.warmth")}: {snap.audio_shaper.warmth_pct.toFixed(1)}%
                    </Text>
                    <Text size="sm">
                      {t("live.masterVol")}: {snap.audio_shaper.master_vol_pct.toFixed(1)}%
                    </Text>
                  </Stack>
                </Card>
              )}

              {snap.tick_budget && (
                <Card withBorder>
                  <Stack gap="xs">
                    <Text fw={500}>{t("live.tickBudget")}</Text>
                    <Text size="sm">
                      {t("live.tickCompute")}: {snap.tick_budget.last_compute_ms.toFixed(1)}
                      {" / "}{snap.tick_budget.budget_ms.toFixed(0)}ms
                    </Text>
                    <Text size="sm">
                      {t("live.tickMax")}: {snap.tick_budget.max_compute_ms.toFixed(1)}ms
                    </Text>
                    <Text
                      size="sm"
                      c={snap.tick_budget.overruns > 0 ? "red" : "dimmed"}
                    >
                      {t("live.tickOverruns")}: {snap.tick_budget.overruns}
                    </Text>
                  </Stack>
                </Card>
              )}

              <PanicButton
                sessionId={snap.session_id}
                disabled={
                  snap.state === "aborted"
                  || snap.state === "completed"
                  || snap.state === "unknown"
                }
              />
            </Stack>
          </Grid.Col>

          <Grid.Col span={12}>
            <Card withBorder>
              <Title order={5}>{t("live.history")}</Title>
              {history.length === 0 ? (
                <Text size="sm" c="dimmed" mt="sm">
                  {t("live.historyEmpty")}
                </Text>
              ) : (
                <TelemetryTimeseries history={history} />
              )}
            </Card>
          </Grid.Col>

          <Grid.Col span={12}>
            <Card withBorder>
              <Title order={5}>{t("live.safetyEvents")}</Title>
              <Stack gap={4} mt="sm">
                {snap.safety_events.length === 0 && (
                  <Text size="sm" c="dimmed">- none -</Text>
                )}
                {snap.safety_events.map((e, i) => (
                  <Text key={i} size="sm">
                    <Badge size="xs" color={e.kind === "panic" ? "red" : "yellow"} mr={6}>
                      {e.kind}
                    </Badge>
                    {e.field}: {e.before.toFixed(2)} → {e.after.toFixed(2)} ({e.reason})
                  </Text>
                ))}
              </Stack>
            </Card>
          </Grid.Col>
        </Grid>
      )}
    </Stack>
  );
}

/**
 * Read-only block listing every creation parameter for this session so the
 * researcher doesn't have to query Mongo to recall what they configured.
 * Telemetry snapshot carries these on every frame; for orphan sessions the
 * daemon hydrates them from Mongo on WS connect.
 */
function SessionDetailsCard({ snap }: { snap: TelemetrySnapshot }) {
  const { t } = useTranslation();
  const started = snap.started_at
    ? new Date(snap.started_at).toLocaleString()
    : "-";
  const audio =
    snap.audio_enabled === true
      ? "on"
      : snap.audio_enabled === false
        ? "off"
        : "auto";
  const params = snap.protocol_params ?? {};
  const hasProtocolParams = Object.keys(params).length > 0;
  return (
    <Card withBorder>
      <Title order={5} mb="sm">{t("live.details")}</Title>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="xs">
        <DetailRow label={t("live.detailsSessionId")} value={
          <Code>{snap.session_id}</Code>
        } />
        <DetailRow label={t("live.detailsSubject")} value={snap.subject_code ?? "-"} />
        <DetailRow label={t("live.detailsStarted")} value={started} />
        <DetailRow label={t("live.detailsProtocol")} value={snap.protocol ?? "-"} />
        <DetailRow label={t("live.detailsEegSource")} value={snap.eeg_source ?? "-"} />
        <DetailRow label={t("live.detailsMaskTarget")} value={snap.mask_target ?? "-"} />
        <DetailRow label={t("live.detailsAudio")} value={audio} />
        <DetailRow label={t("live.detailsFolder")} value={
          snap.folder_path ? <Code>{snap.folder_path}</Code> : "-"
        } />
        <DetailRow label={t("live.detailsTick")} value={`${snap.tick}`} />
      </SimpleGrid>

      {hasProtocolParams && (
        <Stack gap={4} mt="md">
          <Text size="xs" c="dimmed" fw={600} tt="uppercase">
            {t("live.detailsProtocolParams")}
          </Text>
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="xs">
            {Object.entries(params).map(([k, v]) => (
              <DetailRow
                key={k}
                label={k}
                value={
                  v === null || v === undefined
                    ? "-"
                    : typeof v === "string"
                      ? <Code>{v}</Code>
                      : <Code>{JSON.stringify(v)}</Code>
                }
              />
            ))}
          </SimpleGrid>
        </Stack>
      )}
    </Card>
  );
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Group gap={6} wrap="nowrap" align="baseline">
      <Text size="xs" c="dimmed" style={{ minWidth: 110 }}>{label}:</Text>
      <Text size="sm" style={{ wordBreak: "break-all" }}>{value}</Text>
    </Group>
  );
}
