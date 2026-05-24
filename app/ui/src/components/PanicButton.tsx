import { Button } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconAlertOctagon } from "@tabler/icons-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { api } from "../api/client";

export function PanicButton(
  { sessionId, disabled = false }:
  { sessionId: string; disabled?: boolean },
) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const mut = useMutation({
    mutationFn: () => api.panicSession(sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      notifications.show({
        title: "PANIC",
        message: "Session aborted",
        color: "red",
      });
    },
  });
  return (
    <Button
      color="red"
      size="lg"
      leftSection={<IconAlertOctagon size={24} />}
      loading={mut.isPending}
      onClick={() => mut.mutate()}
      disabled={disabled}
      fullWidth
    >
      {t("live.panic")}
    </Button>
  );
}
