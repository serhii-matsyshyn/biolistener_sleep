import { AppShell, Burger, Group, NavLink, Select, Title } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconActivity, IconBulb, IconClipboardList, IconUsers } from "@tabler/icons-react";
import { Link, Outlet, useLocation } from "react-router";
import { useTranslation } from "react-i18next";

export default function App() {
  const { t, i18n } = useTranslation();
  const [opened, { toggle }] = useDisclosure();
  const location = useLocation();

  const navItems = [
    { to: "/live", label: t("app.nav.live"), icon: <IconActivity size={18} /> },
    { to: "/sessions", label: t("app.nav.sessions"), icon: <IconClipboardList size={18} /> },
    { to: "/subjects", label: t("app.nav.subjects"), icon: <IconUsers size={18} /> },
    { to: "/demo-open-loop", label: t("app.nav.demo"), icon: <IconBulb size={18} /> },
  ];

  const onLang = (value: string | null) => {
    if (value) {
      i18n.changeLanguage(value);
      localStorage.setItem("lang", value);
    }
  };

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 220, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Title order={4}>{t("app.title")}</Title>
          </Group>
          <Select
            data={[
              { value: "en", label: "EN" },
              { value: "uk", label: "UK" },
            ]}
            value={i18n.language}
            onChange={onLang}
            allowDeselect={false}
            w={80}
          />
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        {navItems.map((it) => (
          <NavLink
            key={it.to}
            component={Link}
            to={it.to}
            label={it.label}
            leftSection={it.icon}
            active={location.pathname.startsWith(it.to)}
            mb={4}
          />
        ))}
      </AppShell.Navbar>

      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
}
