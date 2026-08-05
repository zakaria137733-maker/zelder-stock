import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import LoginPage from "@/app/login/page";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/api", () => ({
  default: { post: vi.fn() },
}));

import api from "@/lib/api";

const mockedPost = api.post as unknown as ReturnType<typeof vi.fn>;

describe("LoginPage", () => {
  beforeEach(() => {
    mockedPost.mockReset();
    push.mockReset();
    localStorage.clear();
  });

  it("renders the customer sign-in form by default", () => {
    render(<LoginPage />);
    expect(screen.getByText("Welcome back")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Email address")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign In" })).toBeInTheDocument();
  });

  it("stores the token and redirects on successful admin login", async () => {
    mockedPost.mockResolvedValue({ data: { token: "t-123", username: "root" } });
    render(<LoginPage />);

    fireEvent.click(screen.getByRole("button", { name: "Admin sign in" }));
    fireEvent.change(screen.getByPlaceholderText("Admin username"), { target: { value: "root" } });
    fireEvent.change(screen.getByPlaceholderText("Password"), { target: { value: "secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Admin Sign In" }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/"));
    expect(localStorage.getItem("zs_token")).toBe("t-123");
    expect(JSON.parse(localStorage.getItem("zs_user") ?? "{}")).toEqual({ name: "root", role: "admin" });
  });

  it("requires username and password in admin mode without calling the API", () => {
    render(<LoginPage />);
    fireEvent.click(screen.getByRole("button", { name: "Admin sign in" }));
    fireEvent.click(screen.getByRole("button", { name: "Admin Sign In" }));

    expect(screen.getByText("Username and password required")).toBeInTheDocument();
    expect(mockedPost).not.toHaveBeenCalled();
  });

  it("shows the server error detail on a failed login", async () => {
    mockedPost.mockRejectedValue({ response: { data: { detail: "Invalid credentials" } } });
    render(<LoginPage />);

    fireEvent.change(screen.getByPlaceholderText("Email address"), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByPlaceholderText("Password"), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

    expect(await screen.findByText("Invalid credentials")).toBeInTheDocument();
  });

  it("stores token and redirects on successful customer login", async () => {
    mockedPost.mockResolvedValue({ data: { token: "t-456", name: "Alice", email: "a@b.com" } });
    render(<LoginPage />);

    fireEvent.change(screen.getByPlaceholderText("Email address"), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByPlaceholderText("Password"), { target: { value: "pw" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/"));
    expect(JSON.parse(localStorage.getItem("zs_user") ?? "{}")).toEqual({ name: "Alice", email: "a@b.com" });
  });
});
