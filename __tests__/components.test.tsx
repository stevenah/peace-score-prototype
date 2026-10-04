import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { act, render, screen, fireEvent } from "@testing-library/react";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { ColonMap } from "@/components/scoring/ColonMap";
import { VideoUploader } from "@/components/video/VideoUploader";
import { BatchItemCard } from "@/components/video/BatchItemCard";
import { Announcements } from "@/components/workstation/Announcements";
import { CaptureFilmstrip } from "@/components/workstation/CaptureFilmstrip";
import { LeftRail } from "@/components/workstation/LeftRail";
import {
  StationChecklist,
  StationsErrorBoundary,
  nextManualMark,
} from "@/components/workstation/StationChecklist";
import {
  StationGallery,
  StationGalleryOverlay,
} from "@/components/workstation/StationGallery";
import { ProcedureStoreProvider } from "@/hooks/useProcedureMetrics";
import { ProcedureStore } from "@/lib/live/procedure-store";
import type { FrameSample, StationsSlice } from "@/lib/live/procedure-types";
import { STATION_INDEX, STATION_ORDER, STATION_SHORT } from "@/lib/live/stations";
import type { BatchItem, PeaceScore, StationKey } from "@/lib/types";

// -- PeaceScoreCard ---------------------------------------------------------

describe("PeaceScoreCard", () => {
  it("renders score value", () => {
    render(<PeaceScoreCard score={2} />);
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("renders default label from constants", () => {
    render(<PeaceScoreCard score={3} />);
    expect(screen.getByText("Excellent")).toBeInTheDocument();
  });

  it("renders custom label when provided", () => {
    render(<PeaceScoreCard score={1} label="Custom Label" />);
    expect(screen.getByText("Custom Label")).toBeInTheDocument();
  });

  it("renders region name when provided", () => {
    render(<PeaceScoreCard score={2} region="Stomach" />);
    expect(screen.getByText("Stomach")).toBeInTheDocument();
  });

  it("renders all 4 score levels correctly", () => {
    const scores: PeaceScore[] = [0, 1, 2, 3];
    const labels = ["Poor", "Inadequate", "Adequate", "Excellent"];

    scores.forEach((score, i) => {
      const { unmount } = render(<PeaceScoreCard score={score} />);
      expect(screen.getByText(labels[i])).toBeInTheDocument();
      unmount();
    });
  });
});

// -- ColonMap ---------------------------------------------------------------

describe("ColonMap", () => {
  it("renders with default props", () => {
    render(<ColonMap />);
    expect(screen.getByText("Colon Map")).toBeInTheDocument();
    expect(screen.getByText("Segments: 0/8")).toBeInTheDocument();
    expect(screen.getByText("Cecum Not Reached")).toBeInTheDocument();
  });

  it("shows visited segment count", () => {
    render(<ColonMap segmentsVisited={["rectum", "sigmoid", "descending"]} />);
    expect(screen.getByText("Segments: 3/8")).toBeInTheDocument();
  });

  it("shows cecum reached when cecum is visited", () => {
    render(
      <ColonMap
        segmentsVisited={["rectum", "sigmoid", "descending", "splenic_flexure", "transverse", "hepatic_flexure", "ascending", "cecum"]}
      />,
    );
    expect(screen.getByText("Segments: 8/8")).toBeInTheDocument();
    // Check for the checkmark text
    const cecumText = screen.getByText((content) => content.includes("Cecum Reached"));
    expect(cecumText).toBeInTheDocument();
  });

  it("renders segment labels", () => {
    render(<ColonMap />);
    expect(screen.getByText("Rectum")).toBeInTheDocument();
    expect(screen.getByText("Transverse")).toBeInTheDocument();
    expect(screen.getByText("Cecum")).toBeInTheDocument();
  });

  it("renders current position marker when currentSegment is set", () => {
    const { container } = render(<ColonMap currentSegment="transverse" />);
    const circle = container.querySelector("circle");
    expect(circle).not.toBeNull();
    expect(circle?.getAttribute("fill")).toBe("#ef4444");
  });

  it("does not render marker when no current segment", () => {
    const { container } = render(<ColonMap />);
    const circle = container.querySelector("circle");
    expect(circle).toBeNull();
  });
});

// -- VideoUploader ----------------------------------------------------------

describe("VideoUploader", () => {
  it("renders upload prompt", () => {
    render(<VideoUploader onFilesSelect={vi.fn()} />);
    expect(screen.getByText(/drag & drop endoscopy videos/i)).toBeInTheDocument();
    expect(screen.getByText(/MP4, MOV, AVI, MKV/)).toBeInTheDocument();
  });

  it("has correct aria-label", () => {
    render(<VideoUploader onFilesSelect={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /upload endoscopy videos/i }),
    ).toBeInTheDocument();
  });

  it("accepts valid files via input change", () => {
    const onSelect = vi.fn();
    render(<VideoUploader onFilesSelect={onSelect} />);

    const input = document.querySelector("input[type='file']") as HTMLInputElement;
    const file = new File(["video-data"], "test.mp4", { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(onSelect).toHaveBeenCalledWith([file]);
  });

  it("rejects files that are too large", () => {
    const onSelect = vi.fn();
    render(<VideoUploader onFilesSelect={onSelect} />);

    const input = document.querySelector("input[type='file']") as HTMLInputElement;
    // Create a file object with a large size
    const largeFile = new File(["x"], "big.mp4", { type: "video/mp4" });
    Object.defineProperty(largeFile, "size", { value: 1200 * 1024 * 1024 });
    fireEvent.change(input, { target: { files: [largeFile] } });

    // Should not call onFilesSelect with the invalid file
    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByText(/file too large/i)).toBeInTheDocument();
  });

  it("rejects unsupported file types", () => {
    const onSelect = vi.fn();
    render(<VideoUploader onFilesSelect={onSelect} />);

    const input = document.querySelector("input[type='file']") as HTMLInputElement;
    const file = new File(["data"], "doc.pdf", { type: "application/pdf" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByText(/unsupported file type/i)).toBeInTheDocument();
  });

  it("is not interactive when disabled", () => {
    render(<VideoUploader onFilesSelect={vi.fn()} disabled />);
    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("tabindex", "-1");
  });
});

// -- BatchItemCard ----------------------------------------------------------

function makeBatchItem(overrides: Partial<BatchItem> = {}): BatchItem {
  return {
    id: "item-1",
    file: new File(["data"], "test.mp4", { type: "video/mp4" }),
    status: "pending",
    progress: 0,
    analysisId: null,
    analysis: null,
    error: null,
    ...overrides,
  };
}

describe("BatchItemCard", () => {
  it("renders file name", () => {
    render(<BatchItemCard item={makeBatchItem()} onRemove={vi.fn()} />);
    expect(screen.getByText("test.mp4")).toBeInTheDocument();
  });

  it("shows pending status badge", () => {
    render(<BatchItemCard item={makeBatchItem()} onRemove={vi.fn()} />);
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("shows uploading status with progress bar", () => {
    const { container } = render(
      <BatchItemCard
        item={makeBatchItem({ status: "uploading", progress: 0.3 })}
        onRemove={vi.fn()}
      />,
    );
    expect(screen.getByText("Uploading")).toBeInTheDocument();
    // Progress bar should be present
    expect(container.querySelector("[role='progressbar']")).not.toBeNull();
  });

  it("shows completed status with score", () => {
    render(
      <BatchItemCard
        item={makeBatchItem({
          status: "completed",
          progress: 1,
          analysisId: "analysis-1",
          analysis: {
            analysis_id: "analysis-1",
            status: "completed",
            progress: 1,
            created_at: "2025-01-01",
            results: {
              peace_scores: {
                overall: { score: 2, label: "Adequate", confidence: 0.85 },
                by_region: {},
              },
              motion_analysis: { segments: [] },
              timeline: [],
            },
          },
        })}
        onRemove={vi.fn()}
      />,
    );
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("shows error text for failed items", () => {
    render(
      <BatchItemCard
        item={makeBatchItem({ status: "failed", error: "Upload timed out" })}
        onRemove={vi.fn()}
      />,
    );
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Upload timed out")).toBeInTheDocument();
  });

  it("shows remove button for pending/completed/failed", () => {
    const onRemove = vi.fn();
    render(<BatchItemCard item={makeBatchItem({ status: "pending" })} onRemove={onRemove} />);

    const removeBtn = screen.getAllByRole("button").find((el) =>
      el.querySelector("svg"),
    );
    expect(removeBtn).toBeDefined();
    fireEvent.click(removeBtn!);
    expect(onRemove).toHaveBeenCalledWith("item-1");
  });

  it("hides remove button for active states", () => {
    const { container } = render(
      <BatchItemCard
        item={makeBatchItem({ status: "uploading", progress: 0.2 })}
        onRemove={vi.fn()}
      />,
    );
    // The remove button should not be present for uploading state
    const buttons = container.querySelectorAll("button[type='button']");
    expect(buttons.length).toBe(0);
  });
});

// -- ESGE stations -----------------------------------------------------------

function stationsSlice(overrides: Partial<StationsSlice> = {}): StationsSlice {
  const status = overrides.status ?? Array(10).fill("unseen");
  const manual = overrides.manual ?? Array(10).fill(null);
  const observed =
    overrides.observed ??
    status.map((st, i) =>
      manual[i] === "confirmed" ? true : manual[i] === "rejected" ? false : st === "observed",
    );
  return {
    availability: "ok",
    display: true,
    visible: true,
    modelVersion: "lm-1.0.0",
    current: null,
    autoEnabled: Array(10).fill(true),
    observedAtT: Array(10).fill(null),
    total: 10,
    exitingMissing: [],
    ...overrides,
    status,
    manual,
    observed,
    observedCount: observed.filter(Boolean).length,
  };
}

function stationStatus(observed: StationKey[], candidates: StationKey[] = []) {
  return STATION_ORDER.map((k) =>
    observed.includes(k) ? "observed" : candidates.includes(k) ? "candidate" : "unseen",
  ) as StationsSlice["status"];
}

describe("StationChecklist", () => {
  it("renders ten rows in ESGE order, grouped esophagus / duodenum / stomach", () => {
    render(<StationChecklist stations={stationsSlice()} />);
    const rows = screen.getAllByRole("button");
    expect(rows).toHaveLength(10);
    expect(rows[0]).toHaveAccessibleName("Prox. esophagus, Proximal esophagus — not yet observed");
    expect(rows[9]).toHaveAccessibleName(
      "Corpus, Gastric corpus (greater curvature) — not yet observed",
    );
    const groups = screen.getAllByRole("list").map((l) => l.getAttribute("aria-label"));
    expect(groups).toEqual(["Esophagus", "Duodenum", "Stomach"]);
    expect(
      screen.getByText("AI-observed views — not a record of photo documentation."),
    ).toBeInTheDocument();
  });

  it("names each state, including the observation time", () => {
    const observedAtT = Array(10).fill(null);
    observedAtT[STATION_INDEX.antrum] = 192;
    observedAtT[STATION_INDEX.incisura] = 30;
    const manual = Array(10).fill(null);
    manual[STATION_INDEX.incisura] = "confirmed";
    manual[STATION_INDEX.z_line] = "rejected";
    render(
      <StationChecklist
        stations={stationsSlice({
          status: stationStatus(["antrum", "z_line"], ["corpus_greater_curvature"]),
          manual,
          observedAtT,
        })}
      />,
    );
    expect(screen.getByRole("button", { name: "Antrum — observed at 03:12" })).toBeInTheDocument();
    expect(screen.getByText("3:12")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Incisura angularis — confirmed manually" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Z-line — rejected manually" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(
      screen.getByRole("button", {
        name: "Corpus, Gastric corpus (greater curvature) — candidate, not yet observed",
      }),
    ).toHaveAttribute("aria-pressed", "false");

    const glyph = (name: string) =>
      screen.getByRole("button", { name }).querySelector("svg")?.getAttribute("data-glyph");
    expect(glyph("Antrum — observed at 03:12")).toBe("observed");
    expect(glyph("Incisura angularis — confirmed manually")).toBe("confirmed");
    expect(glyph("Z-line — rejected manually")).toBe("rejected");
    expect(glyph("Bulb, Duodenal bulb — not yet observed")).toBe("unseen");
    expect(glyph("Corpus, Gastric corpus (greater curvature) — candidate, not yet observed")).toBe(
      "candidate",
    );
  });

  it("marks the current station and manual-only stations", () => {
    const autoEnabled = Array(10).fill(true);
    autoEnabled[STATION_INDEX.lesser_curvature_retroflex] = false;
    render(
      <StationChecklist stations={stationsSlice({ current: "antrum", autoEnabled })} />,
    );
    expect(screen.getByRole("button", { current: "step" })).toHaveAccessibleName(
      "Antrum — not yet observed",
    );
    expect(
      screen.getByRole("button", {
        name: "Lesser curve, Lesser curvature (partial inversion) — not yet observed, manual only",
      }),
    ).toHaveTextContent("manual");
  });

  it("starts every row's name with its visible label (WCAG 2.5.3 Label in Name)", () => {
    const autoEnabled = Array(10).fill(false);
    render(<StationChecklist stations={stationsSlice({ autoEnabled })} />);
    const rows = screen.getAllByRole("button");
    STATION_ORDER.forEach((key, i) => {
      const name = rows[i].getAttribute("aria-label") ?? "";
      expect(name.startsWith(STATION_SHORT[key])).toBe(true);
      // The visible "manual" chip is spoken too.
      expect(name).toContain("manual");
    });
    // Voice control: "click D2" / "click Lesser curve" must find the row.
    expect(screen.getByRole("button", { name: /^D2\b/ })).toBe(rows[STATION_INDEX.duodenum_descending]);
    expect(screen.getByRole("button", { name: /^Lesser curve\b/ })).toBe(
      rows[STATION_INDEX.lesser_curvature_retroflex],
    );
  });

  it("reports the pressed row's ESGE index", () => {
    const onToggle = vi.fn();
    render(<StationChecklist stations={stationsSlice()} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("button", { name: "Antrum — not yet observed" }));
    expect(onToggle).toHaveBeenCalledWith(STATION_INDEX.antrum);
  });

  it("cycles manual marks: model -> confirmed -> rejected -> model", () => {
    expect(nextManualMark(null)).toBe("confirmed");
    expect(nextManualMark("confirmed")).toBe("rejected");
    expect(nextManualMark("rejected")).toBeNull();
  });

  it("flags a mock model as SIMULATED", () => {
    const { rerender } = render(
      <StationChecklist stations={stationsSlice({ modelVersion: "mock-flow-1" })} />,
    );
    expect(screen.getByText("SIMULATED")).toBeInTheDocument();
    rerender(<StationChecklist stations={stationsSlice({ modelVersion: "lm-1.0.0" })} />);
    expect(screen.queryByText("SIMULATED")).toBeNull();
  });

  it("says so, in one line, for an unsupported video source", () => {
    render(
      <StationChecklist
        stations={stationsSlice({ availability: "unsupported_layout", visible: false })}
      />,
    );
    expect(
      screen.getByText("Station tracking unavailable for this video source"),
    ).toBeInTheDocument();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });

  it("renders nothing when not visible: feature off or shadow mode", () => {
    const off = render(
      <StationChecklist stations={stationsSlice({ availability: "absent", display: false, visible: false })} />,
    );
    expect(off.container).toBeEmptyDOMElement();
    off.unmount();

    const shadow = render(
      <StationChecklist stations={stationsSlice({ display: false, visible: false })} />,
    );
    expect(shadow.container).toBeEmptyDOMElement();
    shadow.unmount();

    const shadowUnsupported = render(
      <StationChecklist
        stations={stationsSlice({
          availability: "unsupported_layout",
          display: false,
          visible: false,
        })}
      />,
    );
    expect(shadowUnsupported.container).toBeEmptyDOMElement();
  });
});

function landmarkFrame(
  t: number,
  opts: { observed?: StationKey[]; current?: StationKey | null; display?: boolean } = {},
): FrameSample {
  return {
    seq: 0,
    t,
    score: 2,
    scoreConfidence: 0.9,
    expectedScore: null,
    region: "stomach",
    motion: "stationary",
    motionConfidence: 0.9,
    frameIndex: 0,
    processingTimeMs: 20,
    landmarkStatus: "ok",
    landmarkTop: opts.current ?? null,
    landmarkModelVersion: "lm-1.0.0",
    stationsDisplay: opts.display ?? true,
    stationCurrent: opts.current ?? null,
    stationStatus: stationStatus(opts.observed ?? []),
    stationAutoEnabled: Array(10).fill(true),
  };
}

function peaceFrame(t: number): FrameSample {
  const { seq, score, scoreConfidence, expectedScore, region, motion, motionConfidence, frameIndex, processingTimeMs } =
    landmarkFrame(t);
  return { seq, t, score, scoreConfidence, expectedScore, region, motion, motionConfidence, frameIndex, processingTimeMs };
}

function renderRail(store: ProcedureStore) {
  return render(
    <ProcedureStoreProvider store={store}>
      <LeftRail dimmed={false} />
    </ProcedureStoreProvider>,
  );
}

describe("LeftRail with stations", () => {
  it("keeps 'Frames analysed' as the hero without landmarks", () => {
    const store = new ProcedureStore();
    store.ingest(peaceFrame(0));
    renderRail(store);
    expect(screen.getByText("Frames analysed:")).toBeInTheDocument();
    expect(screen.queryByText("Stations observed:")).toBeNull();
    expect(screen.queryByRole("region", { name: "ESGE station checklist" })).toBeNull();
  });

  it("swaps the hero to 'Stations observed' N/10 when stations are visible", () => {
    const store = new ProcedureStore();
    store.ingest(landmarkFrame(0, { observed: ["antrum", "incisura"], current: "antrum" }));
    renderRail(store);
    const hero = screen.getByText("Stations observed:").parentElement!;
    expect(hero).toHaveTextContent(/^Stations observed:2\s*\/ 10/);
    // "Frames analysed" moves from the hero to the micro line.
    expect(screen.queryByText("Frames analysed:")).toBeNull();
    expect(hero).toHaveTextContent("Frames analysed 1");
    expect(screen.getByRole("region", { name: "ESGE station checklist" })).toBeInTheDocument();
  });

  it("shows nothing station-related in shadow mode", () => {
    const store = new ProcedureStore();
    store.ingest(landmarkFrame(0, { observed: ["antrum"], display: false }));
    const { container } = renderRail(store);
    expect(screen.getByText("Frames analysed:")).toBeInTheDocument();
    expect(screen.queryByText("Stations observed:")).toBeNull();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    // The tract map is still drawn, but without a single station pin.
    expect(screen.getByRole("img", { name: /^Coverage:/ })).toBeInTheDocument();
    expect(container.querySelector("[data-station]")).toBeNull();
  });

  it("pins all ten stations on the tract map, in their checklist state", () => {
    const store = new ProcedureStore();
    store.ingest(landmarkFrame(0, { observed: ["antrum"], current: "antrum" }));
    const { container } = renderRail(store);
    const pin = (key: StationKey) =>
      container.querySelector(`[data-station="${key}"]`)?.getAttribute("data-state");

    expect(container.querySelectorAll("[data-station]")).toHaveLength(10);
    expect(pin("antrum")).toBe("observed");
    expect(pin("incisura")).toBe("unseen");

    // A manual mark on the row moves the pin with it.
    fireEvent.click(screen.getByRole("button", { name: /^Incisura angularis — not yet observed/ }));
    expect(pin("incisura")).toBe("confirmed");
  });

  it("derives the region label from the current station", () => {
    const store = new ProcedureStore();
    // PEACE says stomach, the station classifier says Z-line.
    store.ingest(landmarkFrame(0, { current: "z_line" }));
    renderRail(store);
    expect(screen.getByText("Esophagus:")).toBeInTheDocument();
    expect(screen.queryByText("Stomach:")).toBeNull();
  });

  it("writes a manual mark to the store when a row is pressed", () => {
    const store = new ProcedureStore();
    store.ingest(landmarkFrame(0, { observed: ["antrum"] }));
    renderRail(store);
    fireEvent.click(screen.getByRole("button", { name: /^Antrum — observed/ }));
    expect(store.getSnapshot().stations.manual[STATION_INDEX.antrum]).toBe("confirmed");
    fireEvent.click(screen.getByRole("button", { name: "Antrum — confirmed manually" }));
    expect(store.getSnapshot().stations.manual[STATION_INDEX.antrum]).toBe("rejected");
    expect(screen.getByRole("button", { name: "Antrum — rejected manually" })).toBeInTheDocument();
  });
});

describe("StationGallery", () => {
  it("shows a 'No frame kept' placeholder for every station without a frame", () => {
    render(<StationGallery stations={stationsSlice()} frames={{}} onClose={vi.fn()} />);
    expect(screen.getAllByText("No frame kept")).toHaveLength(10);
    expect(screen.getAllByRole("listitem")).toHaveLength(10);
  });

  it("shows kept frames in ESGE order with their state", () => {
    render(
      <StationGallery
        stations={stationsSlice({ status: stationStatus(["antrum"]) })}
        frames={{ antrum: { url: "blob:antrum", videoTime: 192, ticket: 3 } }}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getAllByText("No frame kept")).toHaveLength(9);
    const img = screen.getByRole("img", { name: "Antrum, frame at 3:12" });
    expect(img).toHaveAttribute("src", "blob:antrum");
    expect(screen.getAllByRole("listitem")[5]).toContainElement(img);
    expect(screen.getByText("1 / 10 observed")).toBeInTheDocument();
  });

  it("closes", () => {
    const onClose = vi.fn();
    render(<StationGallery stations={stationsSlice()} frames={{}} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: "Close station gallery" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("the connected overlay stays hidden unless open AND stations are visible", () => {
    const store = new ProcedureStore();
    store.ingest(landmarkFrame(0, { display: false }));
    const { rerender } = render(
      <ProcedureStoreProvider store={store}>
        <StationGalleryOverlay open frames={{}} onClose={vi.fn()} />
      </ProcedureStoreProvider>,
    );
    expect(screen.queryByRole("region", { name: "Station gallery" })).toBeNull();

    const visible = new ProcedureStore();
    visible.ingest(landmarkFrame(0));
    rerender(
      <ProcedureStoreProvider store={visible}>
        <StationGalleryOverlay open frames={{}} onClose={vi.fn()} />
      </ProcedureStoreProvider>,
    );
    expect(screen.getByRole("region", { name: "Station gallery" })).toBeInTheDocument();
  });
});

describe("CaptureFilmstrip station chip", () => {
  it("labels a station-observed capture with its station", () => {
    render(
      <CaptureFilmstrip
        captures={[
          {
            id: "3",
            videoTime: 75,
            url: "data:image/jpeg;base64,",
            reason: "station-observed",
            region: "stomach",
            score: 3,
            station: "incisura",
          },
        ]}
      />,
    );
    expect(screen.getByText("Incisura")).toBeInTheDocument();
    expect(screen.getByText("Incisura angularis observed at 1:15")).toBeInTheDocument();
  });
});

describe("Announcements — station count", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function renderAnnouncements(store: ProcedureStore) {
    return render(
      <ProcedureStoreProvider store={store}>
        <Announcements />
      </ProcedureStoreProvider>,
    );
  }

  it("announces the observed count politely, once it settles", () => {
    const store = new ProcedureStore();
    renderAnnouncements(store);
    act(() => {
      store.ingest(landmarkFrame(0, { observed: ["antrum"] }));
      store.ingest(landmarkFrame(0.5, { observed: ["antrum", "incisura"] }));
    });
    expect(screen.queryByText("2 of 10 stations observed")).toBeNull();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    const region = screen.getByText("2 of 10 stations observed");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(screen.queryByText("1 of 10 stations observed")).toBeNull();
  });

  it("says nothing in shadow mode", () => {
    const store = new ProcedureStore();
    renderAnnouncements(store);
    act(() => {
      store.ingest(landmarkFrame(0, { observed: ["antrum"], display: false }));
      vi.advanceTimersByTime(2000);
    });
    expect(screen.queryByText(/stations observed/)).toBeNull();
  });
});

describe("StationsErrorBoundary", () => {
  it("hides a failing station UI instead of taking the page down", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const Boom = () => {
      throw new Error("boom");
    };
    const { container } = render(
      <div>
        <span>rail</span>
        <StationsErrorBoundary>
          <Boom />
        </StationsErrorBoundary>
      </div>,
    );
    expect(container).toHaveTextContent("rail");
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    error.mockRestore();
  });
});
