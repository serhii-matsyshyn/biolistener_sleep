import {
  Button,
  Card,
  Group,
  Modal,
  Select,
  Stack,
  Table,
  TextInput,
  Title,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { api, type AudioMode, type SessionProtocol } from "../api/client";

export default function SessionsRoute() {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [opened, { open, close }] = useDisclosure();
  const [subjectCode, setSubjectCode] = useState("");
  const [audioMode, setAudioMode] = useState<AudioMode>("auto");
  const [protocol, setProtocol] = useState<SessionProtocol>("rl_agent");
  const [olActivity, setOlActivity] = useState<string | null>(null);
  // Track selection by variant_key (always non-null), not by duration_min -
  // some activities ("Mind Games / *", "Instructional Series / *", …) only
  // have a single variant whose duration is null.
  const [olVariantKey, setOlVariantKey] = useState<string | null>(null);
  const [eegSource, setEegSource] = useState("sim");
  const [maskTarget, setMaskTarget] = useState("virtual");

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
  });

  const activitiesQ = useQuery({
    queryKey: ["open-loop-track-activities"],
    queryFn: () => api.listOpenLoopTrackActivities(),
    enabled: protocol === "open_loop_track" && opened,
    retry: false,
  });

  const durationsQ = useQuery({
    queryKey: ["open-loop-track-durations", olActivity],
    queryFn: () => api.listOpenLoopTrackDurations(olActivity!),
    enabled: protocol === "open_loop_track" && !!olActivity,
    retry: false,
  });

  // Auto-select the only variant when the activity has just one - same UX
  // as the Demo Open Loop page, so the user doesn't sit on an empty Duration
  // dropdown for activities like "Mind Games / Love".
  useEffect(() => {
    const variants = durationsQ.data;
    if (!variants || variants.length === 0) return;
    if (variants.length === 1 && olVariantKey !== variants[0].variant_key) {
      setOlVariantKey(variants[0].variant_key);
    }
  }, [durationsQ.data, olVariantKey]);

  const protocolParams = useMemo(() => {
    if (protocol !== "open_loop_track") return undefined;
    const variant = durationsQ.data?.find(
      (v) => v.variant_key === olVariantKey,
    );
    return {
      activity: olActivity,
      duration_min: variant?.duration_min ?? null,
      variant_key: variant?.variant_key ?? null,
      audio_path: variant?.audio_path,
      strobe_path: variant?.strobe_path,
    };
  }, [protocol, olActivity, olVariantKey, durationsQ.data]);

  const audioEnabled: boolean | null =
    audioMode === "on" ? true : audioMode === "off" ? false : null;

  const createMut = useMutation({
    mutationFn: () =>
      api.createSession({
        subject_code: subjectCode,
        protocol,
        protocol_params: protocolParams,
        eeg_source: eegSource,
        mask_target: maskTarget,
        audio_enabled: audioEnabled,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      close();
    },
  });

  const startMut = useMutation({
    mutationFn: (id: string) => api.startSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  const endMut = useMutation({
    mutationFn: (id: string) => api.endSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("sessions.list")}</Title>
        <Button onClick={open}>{t("sessions.newSession")}</Button>
      </Group>

      <Card withBorder>
        <Table verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>id</Table.Th>
              <Table.Th>subject</Table.Th>
              <Table.Th>audio</Table.Th>
              <Table.Th>protocol</Table.Th>
              <Table.Th>status</Table.Th>
              <Table.Th>started</Table.Th>
              <Table.Th>actions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sessionsQ.data?.map((s) => (
              <Table.Tr key={s._id}>
                <Table.Td>
                  <code>{s._id?.slice(0, 8)}</code>
                </Table.Td>
                <Table.Td>{s.subject_id}</Table.Td>
                <Table.Td>
                  {s.audio_enabled === true
                    ? "on"
                    : s.audio_enabled === false
                      ? "off"
                      : "auto"}
                </Table.Td>
                <Table.Td>{s.protocol ?? "rl_agent"}</Table.Td>
                <Table.Td>{s.status}</Table.Td>
                <Table.Td>{new Date(s.started_at).toLocaleString()}</Table.Td>
                <Table.Td>
                  <Group gap="xs">
                    {s.status === "running" ? (
                      <Button
                        size="xs"
                        color="red"
                        onClick={() => endMut.mutate(s._id!)}
                      >
                        {t("sessions.end")}
                      </Button>
                    ) : (
                      // Sessions in a terminal state (completed/aborted) can
                      // never re-enter `running` - the daemon's start_session
                      // rejects anything not in `idle`. Disable the button so
                      // the UI matches the backend contract.
                      <Button
                        size="xs"
                        variant={s.status === "idle" ? "filled" : "default"}
                        disabled={s.status !== "idle"}
                        onClick={() => startMut.mutate(s._id!)}
                      >
                        {t("sessions.start")}
                      </Button>
                    )}
                    <Button
                      size="xs"
                      variant="subtle"
                      component={Link}
                      to={`/live/${s._id}`}
                    >
                      {t("sessions.open")}
                    </Button>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Card>

      <Modal
        opened={opened}
        onClose={close}
        title={t("sessions.newSession")}
      >
        <Stack gap="sm">
          <TextInput
            label={t("sessions.subjectCode")}
            value={subjectCode}
            onChange={(e) => setSubjectCode(e.currentTarget.value)}
            required
          />
          <Select
            label={t("sessions.audio")}
            description={t("sessions.audioHelp")}
            data={[
              { value: "auto", label: t("sessions.audioAuto") },
              { value: "on", label: t("sessions.audioOn") },
              { value: "off", label: t("sessions.audioOff") },
            ]}
            value={audioMode}
            onChange={(v) => v && setAudioMode(v as AudioMode)}
          />
          <Select
            label={t("sessions.protocol")}
            description={t("sessions.protocolHelp")}
            data={[
              { value: "rl_agent", label: t("sessions.protoRl") },
              { value: "baseline_no_stim", label: t("sessions.protoBaseline") },
              { value: "open_loop_track", label: t("sessions.protoOpenLoop") },
            ]}
            value={protocol}
            onChange={(v) => v && setProtocol(v as SessionProtocol)}
          />
          {protocol === "open_loop_track" && (
            <>
              <Select
                label={t("sessions.activity")}
                data={
                  activitiesQ.data?.map((a) => ({
                    value: a.name, label: `${a.block} / ${a.display_name}`,
                  })) ?? []
                }
                value={olActivity}
                onChange={(v) => {
                  setOlActivity(v);
                  setOlVariantKey(null);
                }}
                searchable
                disabled={activitiesQ.isLoading}
                error={
                  activitiesQ.error instanceof Error
                    ? activitiesQ.error.message
                    : activitiesQ.error
                      ? String(activitiesQ.error)
                      : undefined
                }
              />
              <Select
                label={t("sessions.duration")}
                description={
                  (durationsQ.data?.length ?? 0) === 1
                    ? t("sessions.durationSingle")
                    : undefined
                }
                data={
                  durationsQ.data?.map((v) => ({
                    value: v.variant_key,
                    label: v.duration_min !== null
                      ? `${v.duration_min} min`
                      : v.variant_key,
                  })) ?? []
                }
                value={olVariantKey}
                onChange={(v) => setOlVariantKey(v)}
                disabled={
                  !olActivity
                  || durationsQ.isLoading
                  || (durationsQ.data?.length ?? 0) <= 1
                }
              />
            </>
          )}
          <Select
            label={t("sessions.eegSource")}
            description={t("sessions.eegSourceHelp")}
            data={[
              { value: "sim", label: t("sessions.eegSim") },
              { value: "biolistener", label: t("sessions.eegBiolistener") },
            ]}
            value={eegSource}
            onChange={(v) => v && setEegSource(v)}
          />
          <Select
            label={t("sessions.maskTarget")}
            description={t("sessions.maskTargetHelp")}
            data={[
              { value: "virtual", label: t("sessions.maskVirtual") },
              { value: "esp32", label: t("sessions.maskEsp32") },
            ]}
            value={maskTarget}
            onChange={(v) => v && setMaskTarget(v)}
          />
          <Button
            onClick={() => createMut.mutate()}
            loading={createMut.isPending}
            disabled={
              !subjectCode ||
              (protocol === "open_loop_track" && (!olActivity || !olVariantKey))
            }
          >
            {t("sessions.create")}
          </Button>
        </Stack>
      </Modal>
    </Stack>
  );
}
