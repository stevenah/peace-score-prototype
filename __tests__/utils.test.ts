import { describe, it, expect } from "vitest";
import {
  formatDuration,
  formatConfidence,
  formatTimestamp,
  formatFileSize,
} from "@/lib/utils";

describe("formatDuration", () => {
  it("formats 0 seconds", () => {
    expect(formatDuration(0)).toBe("0:00");
  });
  it("formats seconds under a minute", () => {
    expect(formatDuration(45)).toBe("0:45");
  });
  it("formats minutes and seconds", () => {
    expect(formatDuration(125)).toBe("2:05");
  });
  it("floors fractional seconds", () => {
    expect(formatDuration(90.7)).toBe("1:30");
  });
});

describe("formatConfidence", () => {
  it("formats 1.0 as 100%", () => {
    expect(formatConfidence(1)).toBe("100%");
  });
  it("formats 0.85 as 85%", () => {
    expect(formatConfidence(0.85)).toBe("85%");
  });
  it("rounds to nearest percent", () => {
    expect(formatConfidence(0.876)).toBe("88%");
  });
});

describe("formatTimestamp", () => {
  it("formats 0 seconds", () => {
    expect(formatTimestamp(0)).toBe("0:00.0");
  });
  it("formats fractional seconds", () => {
    expect(formatTimestamp(65.3)).toBe("1:05.3");
  });
});

describe("formatFileSize", () => {
  it("formats bytes", () => {
    expect(formatFileSize(500)).toBe("500 B");
  });
  it("formats kilobytes", () => {
    expect(formatFileSize(2048)).toBe("2.0 KB");
  });
  it("formats megabytes", () => {
    expect(formatFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
  });
  it("formats gigabytes", () => {
    expect(formatFileSize(2.5 * 1024 * 1024 * 1024)).toBe("2.5 GB");
  });
});
