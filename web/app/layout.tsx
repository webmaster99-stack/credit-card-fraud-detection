import type { Metadata } from "next";
import Link from "next/link";

import { ApiStatusBanner } from "@/components/ApiStatusBanner";
import "./globals.css";

export const metadata: Metadata = {
  title: "Credit card fraud detector",
  description: "Score a simulated card transaction against the fraud classifier API.",
};

const NAV = [
  { href: "/", label: "Single transaction" },
  { href: "/batch", label: "Batch CSV" },
  { href: "/model", label: "Model info" },
  { href: "/monitoring", label: "Monitoring" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <ApiStatusBanner />
        <header className="border-b border-slate-200 dark:border-slate-800">
          <nav className="mx-auto flex max-w-3xl flex-wrap gap-4 px-4 py-3 text-sm">
            {NAV.map((item) => (
              <Link key={item.href} href={item.href} className="hover:underline">
                {item.label}
              </Link>
            ))}
          </nav>
        </header>
        <main className="mx-auto max-w-3xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
