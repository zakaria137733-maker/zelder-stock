import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import Customers from "@/components/Customers";

vi.mock("@/lib/api", () => ({
  default: { post: vi.fn() },
  getCustomers: vi.fn(),
}));

import { getCustomers } from "@/lib/api";

const mockedGetCustomers = getCustomers as unknown as ReturnType<typeof vi.fn>;

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("Customers", () => {
  it("renders customer rows from the API", async () => {
    mockedGetCustomers.mockResolvedValue([
      {
        id: "c1",
        name: "Sarah Chen",
        email: "sarah@example.com",
        portfolio_value: 50000,
        sentiment_score: 72,
        risk_profile: "aggressive",
        watchlist: ["AAPL", "NVDA"],
        created_at: "2026-01-15T00:00:00Z",
      },
    ]);
    renderWithClient(<Customers />);
    expect(await screen.findByText("Sarah Chen")).toBeInTheDocument();
    expect(screen.getByText("sarah@example.com")).toBeInTheDocument();
    expect(screen.getByText("aggressive")).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();
  });

  it("shows the admin-access notice when the API returns 401", async () => {
    mockedGetCustomers.mockRejectedValue({ response: { status: 401 } });
    renderWithClient(<Customers />);
    expect(await screen.findByText("Admin access required")).toBeInTheDocument();
  });
});
