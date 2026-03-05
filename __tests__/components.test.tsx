import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { ColonMap } from "@/components/scoring/ColonMap";
import { VideoUploader } from "@/components/video/VideoUploader";
import { BatchItemCard } from "@/components/video/BatchItemCard";
import type { BatchItem, PeaceScore } from "@/lib/types";

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
    Object.defineProperty(largeFile, "size", { value: 600 * 1024 * 1024 });
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
