"use client";

import { usePathname } from "next/navigation";
import { Header } from "./Header";

/**
 * Routes that render edge-to-edge, without the site header or the centred
 * max-width main column.
 *
 * The App Router composes layouts rather than replacing them, so a nested layout
 * cannot remove the root <main> wrapper. Deciding here keeps every other route
 * untouched and avoids moving the whole route tree into a group (which would
 * also force a full reload when navigating across the group boundary).
 */
const FULL_BLEED = [/^\/analyze\/live(\/|$)/];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (FULL_BLEED.some((re) => re.test(pathname ?? ""))) {
    return <>{children}</>;
  }

  return (
    <>
      <Header />
      <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
    </>
  );
}
