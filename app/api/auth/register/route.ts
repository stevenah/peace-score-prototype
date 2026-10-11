import { NextResponse } from "next/server";

// Accounts are invite-only: an administrator creates them on /admin.
export async function POST() {
  return NextResponse.json(
    { error: "Sign-up is closed. Ask an administrator for an account." },
    { status: 403 },
  );
}
