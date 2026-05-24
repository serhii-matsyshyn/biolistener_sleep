import {
  Button,
  Card,
  Checkbox,
  Group,
  Modal,
  Stack,
  Table,
  TextInput,
  Title,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../api/client";

const CONTRAINDICATIONS = [
  "photosensitive_epilepsy",
  "generalized_seizures",
  "stroke",
  "recent_traumatic_brain_injury",
  "severe_migraines",
  "psychotic_disorders",
  "dissociative_episodes",
  "alcohol_in_last_24h",
  "psychoactive_substances",
  "seizure_threshold_lowering_medications",
];

export default function SubjectsRoute() {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [opened, { open, close }] = useDisclosure();
  const [code, setCode] = useState("");
  const [checks, setChecks] = useState<Record<string, boolean>>({});

  const subjectsQ = useQuery({
    queryKey: ["subjects"],
    queryFn: () => api.listSubjects(),
  });

  const createMut = useMutation({
    mutationFn: () =>
      api.createSubject({ code, contraindication_checklist: checks }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["subjects"] });
      close();
    },
  });

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("subjects.list")}</Title>
        <Button onClick={open}>{t("subjects.new")}</Button>
      </Group>

      <Card withBorder>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{t("subjects.code")}</Table.Th>
              <Table.Th>created_at</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {subjectsQ.data?.map((s) => (
              <Table.Tr key={s._id}>
                <Table.Td>{s.code}</Table.Td>
                <Table.Td>
                  {(s as any).created_at &&
                    new Date((s as any).created_at).toLocaleString()}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Card>

      <Modal opened={opened} onClose={close} title={t("subjects.new")} size="lg">
        <Stack gap="sm">
          <TextInput
            label={t("subjects.code")}
            value={code}
            onChange={(e) => setCode(e.currentTarget.value)}
            required
          />
          <Title order={5} mt="sm">
            {t("subjects.safetyChecklist")}
          </Title>
          {CONTRAINDICATIONS.map((q) => (
            <Checkbox
              key={q}
              label={q.replace(/_/g, " ")}
              checked={checks[q] ?? false}
              onChange={(e) =>
                setChecks({ ...checks, [q]: e.currentTarget.checked })
              }
            />
          ))}
          <Button
            onClick={() => createMut.mutate()}
            loading={createMut.isPending}
            disabled={!code}
          >
            {t("subjects.create")}
          </Button>
        </Stack>
      </Modal>
    </Stack>
  );
}
