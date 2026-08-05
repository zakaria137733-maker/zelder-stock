import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import AlertsPanel from "@/components/Alerts";
import type { Alert } from "@/lib/types";

vi.mock("@/lib/api", () => ({ getAlerts: vi.fn() }));

import { getAlerts } from "@/lib/api";

const mockedGetAlerts = getAlerts as unknown as ReturnType<typeof vi.fn>;

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

const sampleAlerts: Alert[] = [
  {
    ticker: "NVDA",
    severity: "high",
    direction: "down",
    shift: -12,
    message: "Sharp negative shift across sources",
    previous_score: 55,
    current_score: 43,
    triggered_at: "2026-08-05T10:00:00Z",
  },
];

describe("AlertsPanel", () => {
  it("shows the empty state when there are no alerts", async () => {
    mockedGetAlerts.mockResolvedValue([]);
    renderWithClient(<AlertsPanel />);
    expect(await screen.findByText("No significant sentiment shifts detected")).toBeInTheDocument();
  });

  it("renders active alerts with ticker and severity", async () => {
    mockedGetAlerts.mockResolvedValue(sampleAlerts);
    renderWithClient(<AlertsPanel />);
    expect(await screen.findByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("high")).toBeInTheDocument();
    expect(screen.getByText("Sharp negative shift across sources")).toBeInTheDocument();
    expect(screen.getByText("1 active")).toBeInTheDocument();
  });
});
