"use client";
import "./globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [qc] = useState(() => new QueryClient({
    defaultOptions: { queries: { staleTime: 30000, retry: 1 } }
  }));
  return (
    <html lang="en">
      <head><title>SentimentIQ</title></head>
      <body>
        <QueryClientProvider client={qc}>{children}</QueryClientProvider>
      </body>
    </html>
  );
}
