import type {
  AnatomicalRegion,
  ManualStationMark,
  StationKey,
  StationStatus,
} from "@/lib/types";

/**
 * ESGE upper-GI photodocumentation stations (Bisschops 2016; ESGE 2025).
 *
 * Mirrors contracts/stations.json, the single source of truth shared with
 * ml-backend/app/ml/landmarks/stations.py. __tests__/stations-contract.test.ts
 * asserts the two agree, so edit the JSON first and this file second.
 *
 * Every per-station array — on the wire, in the store, in saved data — is
 * indexed in this ESGE order.
 */
export const STATION_SCHEMA = "esge10.v1";

export const STATION_ORDER: readonly StationKey[] = [
  "esophagus_proximal",
  "esophagus_distal",
  "z_line",
  "duodenal_bulb",
  "duodenum_descending",
  "antrum",
  "cardia_fundus_retroflex",
  "lesser_curvature_retroflex",
  "incisura",
  "corpus_greater_curvature",
];

export const STATIONS_TOTAL = STATION_ORDER.length;

export const STATION_INDEX: Readonly<Record<StationKey, number>> =
  Object.fromEntries(STATION_ORDER.map((k, i) => [k, i])) as Record<
    StationKey,
    number
  >;

export const STATION_ESGE_NO: Readonly<Record<StationKey, number>> =
  Object.fromEntries(STATION_ORDER.map((k, i) => [k, i + 1])) as Record<
    StationKey,
    number
  >;

export const STATION_LABELS: Readonly<Record<StationKey, string>> = {
  esophagus_proximal: "Proximal esophagus",
  esophagus_distal: "Distal esophagus",
  z_line: "Z-line",
  duodenal_bulb: "Duodenal bulb",
  duodenum_descending: "Descending duodenum",
  antrum: "Antrum",
  cardia_fundus_retroflex: "Cardia & fundus (inversion)",
  lesser_curvature_retroflex: "Lesser curvature (partial inversion)",
  incisura: "Incisura angularis",
  corpus_greater_curvature: "Gastric corpus (greater curvature)",
};

export const STATION_SHORT: Readonly<Record<StationKey, string>> = {
  esophagus_proximal: "Prox. esophagus",
  esophagus_distal: "Dist. esophagus",
  z_line: "Z-line",
  duodenal_bulb: "Bulb",
  duodenum_descending: "D2",
  antrum: "Antrum",
  cardia_fundus_retroflex: "Cardia/fundus",
  lesser_curvature_retroflex: "Lesser curve",
  incisura: "Incisura",
  corpus_greater_curvature: "Corpus",
};

export const STATION_REGION: Readonly<Record<StationKey, AnatomicalRegion>> = {
  esophagus_proximal: "esophagus",
  esophagus_distal: "esophagus",
  z_line: "esophagus",
  duodenal_bulb: "duodenum",
  duodenum_descending: "duodenum",
  antrum: "stomach",
  cardia_fundus_retroflex: "stomach",
  lesser_curvature_retroflex: "stomach",
  incisura: "stomach",
  corpus_greater_curvature: "stomach",
};

/** Stations 1-3. */
export const ESOPHAGEAL: readonly StationKey[] = STATION_ORDER.filter(
  (k) => STATION_REGION[k] === "esophagus",
);

/** Stations 4-10: everything past the Z-line. */
export const GASTRODUODENAL: readonly StationKey[] = STATION_ORDER.filter(
  (k) => STATION_REGION[k] !== "esophagus",
);

/** Checklist groups, in ESGE order. */
export const STATION_GROUPS: readonly {
  readonly region: AnatomicalRegion;
  readonly stations: readonly StationKey[];
}[] = [
  { region: "esophagus", stations: ESOPHAGEAL },
  { region: "duodenum", stations: STATION_ORDER.filter((k) => STATION_REGION[k] === "duodenum") },
  { region: "stomach", stations: STATION_ORDER.filter((k) => STATION_REGION[k] === "stomach") },
];

export function isStationKey(value: unknown): value is StationKey {
  return (
    typeof value === "string" &&
    Object.prototype.hasOwnProperty.call(STATION_INDEX, value)
  );
}

/**
 * Effective "observed" for one station: a clinician's mark wins, otherwise the
 * model's sticky status decides.
 */
export function isObserved(
  status: StationStatus,
  manual: ManualStationMark,
): boolean {
  if (manual === "confirmed") return true;
  if (manual === "rejected") return false;
  return status === "observed";
}

/**
 * Stations in `among` (ESGE order preserved) that are auto-enabled and not yet
 * observed. Manual-only stations are never reported missing: the model cannot
 * observe them, so nagging about them would be noise.
 */
export function missingStations(
  observed: readonly boolean[],
  autoEnabled: readonly boolean[],
  among: readonly StationKey[] = STATION_ORDER,
): StationKey[] {
  return among.filter((k) => {
    const i = STATION_INDEX[k];
    return autoEnabled[i] !== false && !observed[i];
  });
}
