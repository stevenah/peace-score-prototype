import { describe, it, expect } from "vitest";

// Extracted from upload route for testability
function isValidVideoFile(header: Uint8Array): boolean {
  // MP4/MOV: "ftyp" at offset 4
  if (header.length >= 8 && header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70) return true;
  // AVI: "RIFF" header
  if (header.length >= 4 && header[0] === 0x52 && header[1] === 0x49 && header[2] === 0x46 && header[3] === 0x46) return true;
  // MKV/WebM: EBML header
  if (header.length >= 4 && header[0] === 0x1a && header[1] === 0x45 && header[2] === 0xdf && header[3] === 0xa3) return true;
  return false;
}

describe("isValidVideoFile", () => {
  it("accepts MP4 files (ftyp at offset 4)", () => {
    // MP4 header: [size bytes] + "ftyp"
    const header = new Uint8Array([0x00, 0x00, 0x00, 0x1c, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d]);
    expect(isValidVideoFile(header)).toBe(true);
  });

  it("accepts AVI files (RIFF header)", () => {
    const header = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00]);
    expect(isValidVideoFile(header)).toBe(true);
  });

  it("accepts MKV/WebM files (EBML header)", () => {
    const header = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0x00, 0x00, 0x00, 0x00]);
    expect(isValidVideoFile(header)).toBe(true);
  });

  it("rejects PNG files", () => {
    const header = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    expect(isValidVideoFile(header)).toBe(false);
  });

  it("rejects JPEG files", () => {
    const header = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00]);
    expect(isValidVideoFile(header)).toBe(false);
  });

  it("rejects empty buffers", () => {
    expect(isValidVideoFile(new Uint8Array([]))).toBe(false);
  });

  it("rejects too-short buffers", () => {
    expect(isValidVideoFile(new Uint8Array([0x66, 0x74]))).toBe(false);
  });
});
