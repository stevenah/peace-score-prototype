"use client";

import { useCallback, useState } from "react";
import { LiveWorkstationRun } from "./LiveWorkstationRun";
import { PreProcedureEntry, type ProcedureSource } from "./PreProcedureEntry";

interface Session extends ProcedureSource {
  id: string;
}

let sessionCounter = 0;

/**
 * Owns the procedure session and nothing else.
 *
 * Each run is keyed by session id, so starting a second procedure remounts the
 * feed, the store, the filmstrip and the capture bucket. Cross-run contamination
 * becomes structurally impossible rather than depending on remembering to reset —
 * which is exactly the bug the previous implementation had.
 */
export function LiveWorkstation() {
  const [session, setSession] = useState<Session | null>(null);

  const start = useCallback((s: ProcedureSource) => {
    sessionCounter += 1;
    setSession({ ...s, id: `run-${sessionCounter}` });
  }, []);

  const exit = useCallback(() => setSession(null), []);

  return (
    <div
      data-workstation
      className="dark h-dvh w-full overflow-hidden overscroll-none bg-ws-void font-sans text-ws-fg antialiased"
    >
      {session ? (
        <LiveWorkstationRun key={session.id} session={session} onExit={exit} />
      ) : (
        <PreProcedureEntry onStart={start} />
      )}
    </div>
  );
}
