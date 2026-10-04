import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import v1Fixture from "@/contracts/live_frame_result.v1.example.json";
import v2Fixture from "@/contracts/live_frame_result.v2.example.json";
import type { LiveFrameResult } from "@/lib/types";

/**
 * LiveWorkstationRun end to end, minus the socket and the <video>: results are
 * pushed straight into the feed's onResult and the video's callbacks are driven
 * by hand. What is under test is the wiring — tickets, gating, save, PATCH.
 */

type OnResult = (result: LiveFrameResult, videoTime: number, ticket: number) => void;

const feed = vi.hoisted(() => ({
  onResult: null as null | OnResult,
  nextTicket: 0,
}));

vi.mock("@/hooks/useLiveFeed", () => ({
  useLiveFeed: ({ onResult }: { onResult: OnResult }) => {
    feed.onResult = onResult;
    return {
      isConnected: true,
      isConnecting: false,
      connectionError: null,
      inFlightFrames: 0,
      lastResultAtMs: null,
      sendFrame: () => feed.nextTicket++,
    };
  },
}));

type VideoProps = ComponentProps<
  typeof import("@/components/workstation/ProcedureVideo").ProcedureVideo
>;
const video = vi.hoisted(() => ({ props: null as null | VideoProps }));

vi.mock("@/components/workstation/ProcedureVideo", () => ({
  ProcedureVideo: (props: VideoProps) => {
    video.props = props;
    return <div />;
  },
}));

vi.mock("@/lib/api-client", () => ({
  saveLiveAnalysis: vi.fn(),
  updateLiveStations: vi.fn(),
}));

import { LiveWorkstationRun } from "@/components/workstation/LiveWorkstationRun";
import { saveLiveAnalysis, updateLiveStations } from "@/lib/api-client";
import { STATION_INDEX } from "@/lib/live/stations";

const save = vi.mocked(saveLiveAnalysis);
const patch = vi.mocked(updateLiveStations);

const originalCreate = URL.createObjectURL;
const originalRevoke = URL.revokeObjectURL;

beforeEach(() => {
  feed.onResult = null;
  feed.nextTicket = 0;
  video.props = null;
  save.mockReset();
  patch.mockReset();
  save.mockResolvedValue({ id: "db-1", analysisId: "live_1" });
  patch.mockResolvedValue({ stations: {} as never });
  URL.createObjectURL = vi.fn(() => "blob:frame");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  URL.createObjectURL = originalCreate;
  URL.revokeObjectURL = originalRevoke;
});

function renderRun(onExit = vi.fn()) {
  render(
    <LiveWorkstationRun
      session={{ source: "https://example.test/stream.m3u8", mode: "url", label: "run-1" }}
      onExit={onExit}
    />,
  );
  return { onExit };
}

/** One frame through capture -> send -> result. */
function frame(result: object, videoTime: number) {
  act(() => {
    video.props!.onFrameCapture(new Blob(["jpeg"]), "data:image/jpeg;base64,", videoTime);
  });
  const ticket = feed.nextTicket - 1;
  act(() => {
    feed.onResult!(
      { ...(result as LiveFrameResult), frame_index: ticket },
      videoTime,
      ticket,
    );
  });
}

describe("LiveWorkstationRun — stations wiring", () => {
  it("feature off: no station UI, and a save identical to before", async () => {
    renderRun();
    frame(v1Fixture, 1);
    frame(v1Fixture, 1.5);
    expect(screen.getByText("Frames analysed:")).toBeInTheDocument();
    expect(screen.queryByText("Stations observed:")).toBeNull();

    await act(async () => video.props!.onVideoEnd!());
    expect(save).toHaveBeenCalledTimes(1);
    const data = save.mock.calls[0][0];
    expect(data.stations).toBeUndefined();
    expect(data.timeline[0]).toEqual({
      timestamp: 1,
      frame_index: 0,
      motion: "stationary",
      region: "stomach",
      peace_score: 2,
      confidence: 0.81,
    });
    expect(screen.queryByRole("region", { name: "Station gallery" })).toBeNull();
  });

  it("feature on: checklist, station chip, gallery, and a save with stations", async () => {
    renderRun();
    frame(v2Fixture, 1);

    expect(screen.getByText("Stations observed:")).toBeInTheDocument();
    // The filmstrip captured the frame on which Antrum became observed.
    expect(screen.getByText("Antrum observed at 0:01")).toBeInTheDocument();

    await act(async () => video.props!.onVideoEnd!());
    const data = save.mock.calls[0][0];
    expect(data.stations).toMatchObject({
      schema: "esge10.v1",
      model_version: "lm-1.0.0",
      display: true,
      availability: "ok",
    });
    expect(data.timeline[0]).toMatchObject({
      landmark_top: "antrum",
      landmark_status: "ok",
      station_current: "antrum",
    });

    // End-of-procedure gallery, with the antrum frame promoted.
    const gallery = screen.getByRole("region", { name: "Station gallery" });
    expect(gallery).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Antrum, frame at 0:01" })).toHaveAttribute(
      "src",
      "blob:frame",
    );
    fireEvent.click(screen.getByRole("button", { name: "Close station gallery" }));
    expect(screen.queryByRole("region", { name: "Station gallery" })).toBeNull();
  });

  it("PATCHes manual overrides made after the save", async () => {
    renderRun();
    frame(v2Fixture, 1);
    await act(async () => video.props!.onVideoEnd!());
    expect(save).toHaveBeenCalledTimes(1);
    expect(patch).not.toHaveBeenCalled();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Incisura angularis — not yet observed" }));
    });
    expect(patch).toHaveBeenCalledTimes(1);
    const [id, body] = patch.mock.calls[0];
    expect(id).toBe("live_1");
    expect(body.manual[8]).toBe("confirmed");
    expect(body.observed_at_t[8]).toBe(1);
  });

  it("retries a failed override PATCH, and flushes it on exit", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const { onExit } = renderRun();
      frame(v2Fixture, 1);
      await act(async () => video.props!.onVideoEnd!());
      patch.mockRejectedValueOnce(new Error("503"));

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Incisura angularis — not yet observed" }));
      });
      expect(patch).toHaveBeenCalledTimes(1);
      expect(screen.getByText("Save failed")).toBeInTheDocument();

      // A later publish neither drops the mark nor hammers the server...
      frame(v2Fixture, 1.5);
      expect(patch).toHaveBeenCalledTimes(1);
      // ...and the backoff timer retries it without one.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(patch).toHaveBeenCalledTimes(2);
      expect(patch.mock.calls[1][1].manual[8]).toBe("confirmed");
      expect(screen.getByText("Saved to dashboard")).toBeInTheDocument();

      // A second override fails; exit sends it at once rather than losing it.
      patch.mockRejectedValueOnce(new Error("503"));
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Antrum — observed at 00:01" }));
      });
      expect(patch).toHaveBeenCalledTimes(3);
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Exit workstation" }));
      });
      expect(patch).toHaveBeenCalledTimes(4);
      expect(patch.mock.calls[3][1].manual[STATION_INDEX.antrum]).toBe("confirmed");
      expect(save).toHaveBeenCalledTimes(1);
      expect(onExit).toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("a failed save is retried on exit, carrying overrides made since", async () => {
    save.mockRejectedValueOnce(new Error("500"));
    renderRun();
    frame(v2Fixture, 1);
    await act(async () => video.props!.onVideoEnd!());
    expect(save).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Save failed")).toBeInTheDocument();

    // Nothing saved yet, so nothing to PATCH: the retried save carries it.
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Incisura angularis — not yet observed" }));
    });
    expect(patch).not.toHaveBeenCalled();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Exit workstation" }));
    });
    expect(save).toHaveBeenCalledTimes(2);
    expect(save.mock.calls[1][0].stations?.manual[8]).toBe("confirmed");
    expect(patch).not.toHaveBeenCalled();
  });

  it("saves on exit — URL streams never fire onEnded", async () => {
    const { onExit } = renderRun();
    frame(v2Fixture, 1);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Exit workstation" }));
    });
    expect(save).toHaveBeenCalledTimes(1);
    expect(onExit).toHaveBeenCalled();
  });

  it("shadow mode: records stations, shows none of them", async () => {
    renderRun();
    const shadow = { ...v2Fixture, landmark: { ...v2Fixture.landmark, display: false } };
    frame(shadow, 1);
    expect(screen.queryByText("Stations observed:")).toBeNull();
    expect(screen.queryByText("Antrum observed at 0:01")).toBeNull();

    await act(async () => video.props!.onVideoEnd!());
    expect(screen.queryByRole("region", { name: "Station gallery" })).toBeNull();
    expect(save.mock.calls[0][0].stations).toMatchObject({ display: false });
  });

  it("a malformed landmark block costs the stations, not the PEACE sample", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    renderRun();
    frame({ ...v1Fixture, landmark: { schema: "esge10.v1", probs: "junk" } }, 1);
    frame({ ...v1Fixture, landmark: { schema: "esge10.v1", probs: "junk" } }, 1.5);
    expect(screen.getByText("Frames analysed:").nextElementSibling).toHaveTextContent("2");
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });
});
