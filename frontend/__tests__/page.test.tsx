import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import Home from "@/app/page";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/components/Sidebar", () => ({ default: () => null }));
vi.mock("@/components/Topbar", () => ({ default: () => null }));
vi.mock("@/components/Dashboard", () => ({ default: () => null }));
vi.mock("@/components/SentimentFeed", () => ({ default: () => null }));
vi.mock("@/components/MarketData", () => ({ default: () => null }));
vi.mock("@/components/Customers", () => ({ default: () => null }));
vi.mock("@/components/Transactions", () => ({ default: () => null }));

describe("Home auth redirect", () => {
  beforeEach(() => {
    push.mockReset();
    localStorage.clear();
  });

  it("redirects to /login when no token is stored", () => {
    render(<Home />);
    expect(push).toHaveBeenCalledWith("/login");
  });

  it("does not redirect when a token is stored", () => {
    localStorage.setItem("zs_token", "t-123");
    render(<Home />);
    expect(push).not.toHaveBeenCalled();
  });
});
