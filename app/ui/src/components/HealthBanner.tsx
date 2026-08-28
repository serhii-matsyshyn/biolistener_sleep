import { Alert, Code, Text } from "@mantine/core";
import { IconDatabaseOff } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

type Health = {
  mongo: { ok: boolean; url: string; db: string };
  protocol_runner: { ok: boolean };
};

/**
 * Sits above every page while a dependency is down.
 *
 * Without it the failure surfaces as a request that hangs and then returns
 * 500, on whichever page happened to need it, with nothing saying why. The
 * hardware protocol keeps its state in the running process, so it is worth
 * saying which pages are actually affected rather than implying the whole
 * daemon is unusable.
 */
export function HealthBanner() {
  const { t } = useTranslation();
  const { data, isError } = useQuery<Health>({
    queryKey: ["health"],
    queryFn: async () => {
      const r = await fetch("/api/health");
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    },
    refetchInterval: 10_000,
    retry: false,
  });

  if (isError) {
    return (
      <Alert color="red" variant="filled" radius={0} icon={<IconDatabaseOff size={18} />}
             title={t("health.daemonTitle")}>
        {t("health.daemonBody")}
      </Alert>
    );
  }
  if (!data || data.mongo.ok) return null;

  return (
    <Alert color="red" variant="filled" radius={0} icon={<IconDatabaseOff size={18} />}
           title={t("health.mongoTitle")}>
      <Text size="sm">{t("health.mongoBody")}</Text>
      <Text size="sm" mt={4}>
        {t("health.mongoUrl")} <Code c="red.0" bg="red.9">{data.mongo.url}</Code>
      </Text>
    </Alert>
  );
}
