"use client";

import {
  createContext,
  createElement,
  useContext,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { ProcedureStore } from "@/lib/live/procedure-store";
import type { ProcedureSnapshot } from "@/lib/live/procedure-types";

const StoreContext = createContext<ProcedureStore | null>(null);

export function ProcedureStoreProvider({
  store,
  children,
}: {
  store: ProcedureStore;
  children: ReactNode;
}) {
  return createElement(StoreContext.Provider, { value: store }, children);
}

export function useProcedureStore(): ProcedureStore {
  const store = useContext(StoreContext);
  if (!store) {
    throw new Error("useProcedureStore must be used inside a ProcedureStoreProvider");
  }
  return store;
}

/**
 * Subscribe to one slice of the procedure snapshot.
 *
 * The selector MUST return a referentially stable value when nothing it depends
 * on has changed, or React will warn about tearing / loop. That is exactly why
 * the store publishes pre-built per-slice objects and reuses them when unchanged:
 *   OK:  (s) => s.timers          (s) => s.score.display
 *   NOT: (s) => ({ a: s.a, b: s.b })
 */
export function useProcedureSlice<T>(select: (s: ProcedureSnapshot) => T): T {
  const store = useProcedureStore();
  const subscribe = store.subscribe;
  const getSnapshot = useMemo(
    () => () => select(store.getSnapshot()),
    // The selector is expected to be a module-level or stable function.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store],
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export const selectTimers = (s: ProcedureSnapshot) => s.timers;
export const selectRegions = (s: ProcedureSnapshot) => s.regions;
export const selectCoverage = (s: ProcedureSnapshot) => s.coverage;
export const selectScore = (s: ProcedureSnapshot) => s.score;
export const selectVisibility = (s: ProcedureSnapshot) => s.visibility;
export const selectFeed = (s: ProcedureSnapshot) => s.feed;
/** Gate every consumer on `.visible` — see StationsSlice. */
export const selectStations = (s: ProcedureSnapshot) => s.stations;
