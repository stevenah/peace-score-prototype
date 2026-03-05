import { describe, it, expect } from "vitest";
import { getScoreColor, getScoreLabel, cn } from "@/lib/utils";
import {
  PEACE_SCORE_LABELS,
  PEACE_SCORE_COLORS,
  MOTION_LABELS,
  REGION_LABELS,
  REGION_ORDER,
  ALLOWED_VIDEO_TYPES,
  MAX_FILE_SIZE_MB,
  MAX_FILE_SIZE_BYTES,
  getWsUrl,
} from "@/lib/constants";
import type { PeaceScore } from "@/lib/types";

describe("getScoreColor", () => {
  it("returns red for score 0", () => {
    expect(getScoreColor(0)).toBe("#ef4444");
  });
  it("returns orange for score 1", () => {
    expect(getScoreColor(1)).toBe("#f97316");
  });
  it("returns lime for score 2", () => {
    expect(getScoreColor(2)).toBe("#84cc16");
  });
  it("returns green for score 3", () => {
    expect(getScoreColor(3)).toBe("#22c55e");
  });
});

describe("getScoreLabel", () => {
  it("returns correct labels for all scores", () => {
    expect(getScoreLabel(0)).toBe("Poor");
    expect(getScoreLabel(1)).toBe("Inadequate");
    expect(getScoreLabel(2)).toBe("Adequate");
    expect(getScoreLabel(3)).toBe("Excellent");
  });
});

describe("cn utility", () => {
  it("merges class names", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible");
  });

  it("deduplicates tailwind classes", () => {
    expect(cn("p-4", "p-2")).toBe("p-2");
  });

  it("handles undefined and null", () => {
    expect(cn("a", undefined, null, "b")).toBe("a b");
  });
});

describe("constants integrity", () => {
  it("PEACE_SCORE_LABELS has all 4 scores", () => {
    expect(Object.keys(PEACE_SCORE_LABELS)).toHaveLength(4);
    ([0, 1, 2, 3] as PeaceScore[]).forEach((score) => {
      expect(PEACE_SCORE_LABELS[score]).toBeTruthy();
    });
  });

  it("PEACE_SCORE_COLORS has all 4 scores", () => {
    expect(Object.keys(PEACE_SCORE_COLORS)).toHaveLength(4);
    ([0, 1, 2, 3] as PeaceScore[]).forEach((score) => {
      expect(PEACE_SCORE_COLORS[score]).toMatch(/^#[0-9a-f]{6}$/);
    });
  });

  it("MOTION_LABELS covers all directions", () => {
    expect(MOTION_LABELS.insertion).toBe("Inserting");
    expect(MOTION_LABELS.retraction).toBe("Retracting");
    expect(MOTION_LABELS.stationary).toBe("Stationary");
  });

  it("REGION_LABELS covers all regions", () => {
    expect(REGION_LABELS.esophagus).toBe("Esophagus");
    expect(REGION_LABELS.stomach).toBe("Stomach");
    expect(REGION_LABELS.duodenum).toBe("Duodenum");
  });

  it("REGION_ORDER matches expected anatomical order", () => {
    expect(REGION_ORDER).toEqual(["esophagus", "stomach", "duodenum"]);
  });

  it("ALLOWED_VIDEO_TYPES includes common video formats", () => {
    expect(ALLOWED_VIDEO_TYPES).toContain("video/mp4");
    expect(ALLOWED_VIDEO_TYPES).toContain("video/quicktime");
    expect(ALLOWED_VIDEO_TYPES).toContain("video/x-msvideo");
    expect(ALLOWED_VIDEO_TYPES).toContain("video/x-matroska");
  });

  it("MAX_FILE_SIZE_BYTES equals MB * 1024 * 1024", () => {
    expect(MAX_FILE_SIZE_BYTES).toBe(MAX_FILE_SIZE_MB * 1024 * 1024);
  });
});

describe("getWsUrl", () => {
  it("returns fallback URL in node/test environment", () => {
    // In jsdom, window exists but NEXT_PUBLIC_WS_URL is not set
    const url = getWsUrl();
    expect(url).toBe("ws://localhost:8000/api/v1/ws/live");
  });
});
