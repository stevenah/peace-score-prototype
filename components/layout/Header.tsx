"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { LogOut } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ThemeToggle } from "./ThemeToggle";

function getInitials(name?: string | null, email?: string | null): string {
  if (name) {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  }
  return email?.[0]?.toUpperCase() ?? "U";
}

export function Header() {
  const pathname = usePathname();
  const { data: session, status } = useSession();

  const isAuthPage = pathname === "/login" || pathname === "/register";

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/85 shadow-[0_1px_3px_0_rgb(0_0_0/0.04)] backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <svg
            width="28"
            height="28"
            viewBox="0 0 28 28"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="text-primary"
            aria-hidden="true"
          >
            {/* Shield outline */}
            <path
              d="M14 2L4 6.5V13C4 19.35 8.26 25.27 14 27C19.74 25.27 24 19.35 24 13V6.5L14 2Z"
              fill="currentColor"
              fillOpacity="0.12"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
            {/* Heartbeat/waveform line */}
            <path
              d="M7 14.5H10.5L12 11L14 18L16 11L17.5 14.5H21"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-lg font-bold tracking-tight">PEACE</span>
        </Link>

        {!isAuthPage && (status === "loading" || session) && (
          <div className="flex items-center gap-2">
            <ThemeToggle />

            {session && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="flex items-center gap-2 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
                    <Avatar size="sm">
                      <AvatarFallback>
                        {getInitials(session.user?.name, session.user?.email)}
                      </AvatarFallback>
                    </Avatar>
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                      {session.user?.name && (
                        <p className="text-sm font-medium leading-none">
                          {session.user.name}
                        </p>
                      )}
                      {session.user?.email && (
                        <p className="text-xs text-muted-foreground leading-none">
                          {session.user.email}
                        </p>
                      )}
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onClick={() => signOut({ callbackUrl: "/login" })}
                  >
                    <LogOut className="mr-2 h-4 w-4" />
                    Sign out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        )}

        {isAuthPage && (
          <ThemeToggle />
        )}
      </div>
    </header>
  );
}
