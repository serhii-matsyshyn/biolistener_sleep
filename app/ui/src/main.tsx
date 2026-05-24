import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider, createTheme } from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./i18n";
import App from "./App";
import LiveRoute from "./routes/live";
import SessionsRoute from "./routes/sessions";
import SubjectsRoute from "./routes/subjects";
import DemoOpenLoopRoute from "./routes/demo_open_loop";

const theme = createTheme({
  primaryColor: "teal",
  defaultRadius: "md",
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Notifications position="top-right" />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route element={<App />}>
              <Route index element={<Navigate to="/live" replace />} />
              <Route path="live" element={<LiveRoute />} />
              <Route path="live/:sessionId" element={<LiveRoute />} />
              <Route path="sessions" element={<SessionsRoute />} />
              <Route path="subjects" element={<SubjectsRoute />} />
              <Route path="demo-open-loop" element={<DemoOpenLoopRoute />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </MantineProvider>
  </React.StrictMode>,
);
