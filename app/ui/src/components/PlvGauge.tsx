import { Group, Progress, Stack, Text, Tooltip } from "@mantine/core";

interface Props {
  plv: number | null | undefined;
}

// PLV (phase-locking value) is a unit-less coherence in [0, 1]:
//   0.0 = no phase relationship between strobe and EEG (no entrainment)
//   1.0 = perfect tick-by-tick lock
// In practice the band the policy actually drives sits around 0.2-0.5 once
// entrainment kicks in. We render it as a fixed [0, 1] progress bar so the
// reference is the same from one tick to the next - that's how a human eye
// reads "is it improving?" without doing math.
export function PlvGauge({ plv }: Props) {
  const value = plv ?? 0;
  // Visual buckets — colour as a quick "is it good?" signal:
  //   < 0.15  red    → essentially no lock
  //   0.15-0.30 yellow → present but weak
  //   >= 0.30 teal  → meaningful entrainment
  const color = value >= 0.3 ? "teal" : value >= 0.15 ? "yellow" : "red";
  return (
    <Stack gap={4}>
      <Group justify="space-between">
        <Text fw={500}>PLV (entrainment)</Text>
        <Tooltip label="Phase-locking value: 0 = no lock, 1 = perfect lock">
          <Text fw={500} c={color}>{value.toFixed(3)}</Text>
        </Tooltip>
      </Group>
      <Progress value={value * 100} color={color} size="lg" />
      <Group justify="space-between">
        <Text size="xs" c="dimmed">0.0</Text>
        <Text size="xs" c="dimmed">0.5</Text>
        <Text size="xs" c="dimmed">1.0</Text>
      </Group>
    </Stack>
  );
}
