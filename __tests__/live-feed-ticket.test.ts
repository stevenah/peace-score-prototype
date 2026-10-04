import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import v1Fixture from "@/contracts/live_frame_result.v1.example.json";

/**
 * A WebSocket double with a tiny fake server behind it: like the real backend,
 * it numbers frames 0, 1, 2… per connection in the order it receives them.
 */
class MockSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockSocket[] = [];

  readyState = MockSocket.CONNECTING;
  received: ArrayBuffer[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;

  constructor(public url: string) {
    MockSocket.instances.push(this);
  }

  send(data: ArrayBuffer) {
    this.received.push(data);
  }

  close() {
    this.readyState = MockSocket.CLOSED;
  }

  accept() {
    this.readyState = MockSocket.OPEN;
    this.onopen?.();
  }

  /** Reply to the i-th frame this "server" received. */
  reply(frameIndex: number) {
    this.onmessage?.({
      data: JSON.stringify({ ...v1Fixture, frame_index: frameIndex }),
    });
  }
}

const frame = (n = 4) =>
  ({ arrayBuffer: () => Promise.resolve(new ArrayBuffer(n)) }) as unknown as Blob;
const unreadable = () =>
  ({ arrayBuffer: () => Promise.reject(new Error("gone")) }) as unknown as Blob;

/** Let the serialised send chain run to completion. */
const flushSends = () => act(() => new Promise<void>((r) => setTimeout(r, 0)));

async function connectedFeed(onResult = vi.fn()) {
  const hook = renderHook(() => useLiveFeed({ enabled: true, onResult }));
  await act(async () => {}); // /api/live fetch -> new WebSocket
  const ws = MockSocket.instances.at(-1)!;
  act(() => ws.accept());
  return { ...hook, ws, onResult };
}

beforeEach(() => {
  MockSocket.instances = [];
  vi.stubGlobal("WebSocket", MockSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ ws_url: "ws://test/live" }))),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useLiveFeed — send tickets", () => {
  it("returns null when there is no open connection", async () => {
    const hook = renderHook(() => useLiveFeed({ enabled: true }));
    await act(async () => {});
    expect(hook.result.current.sendFrame(frame(), 0)).toBeNull();
  });

  it("hands out tickets that match the server's frame_index", async () => {
    const { result, ws, onResult } = await connectedFeed();
    const tickets: (number | null)[] = [];
    act(() => {
      for (let i = 0; i < 4; i++) tickets.push(result.current.sendFrame(frame(), i * 0.5));
    });
    expect(tickets).toEqual([0, 1, 2, 3]);
    expect(result.current.inFlightFrames).toBe(4);

    await flushSends();
    expect(ws.received).toHaveLength(4);

    act(() => {
      for (let i = 0; i < 4; i++) ws.reply(i);
    });
    expect(onResult).toHaveBeenCalledTimes(4);
    onResult.mock.calls.forEach(([res, videoTime, ticket], i) => {
      expect(res.frame_index).toBe(i);
      expect(ticket).toBe(tickets[i]);
      expect(videoTime).toBe(i * 0.5);
    });
    expect(result.current.inFlightFrames).toBe(0);
  });

  it("does not desync when a frame's arrayBuffer() rejects", async () => {
    const { result, ws, onResult } = await connectedFeed();
    let a = -1;
    let b = -1;
    let c = -1;
    act(() => {
      a = result.current.sendFrame(frame(), 1.0)!;
      b = result.current.sendFrame(unreadable(), 1.5)!;
      c = result.current.sendFrame(frame(), 2.0)!;
    });
    await flushSends();

    // Only two frames reached the server; it numbers them 0 and 1.
    expect(ws.received).toHaveLength(2);
    expect(result.current.inFlightFrames).toBe(2);
    act(() => {
      ws.reply(0);
      ws.reply(1);
    });

    // Server index 1 is the THIRD frame: its own ticket and video time, not the
    // failed frame's.
    expect(onResult).toHaveBeenCalledTimes(2);
    expect(onResult.mock.calls[0].slice(1)).toEqual([1.0, a]);
    expect(onResult.mock.calls[1].slice(1)).toEqual([2.0, c]);
    expect(onResult.mock.calls.some((call) => call[2] === b)).toBe(false);
    expect(result.current.inFlightFrames).toBe(0);

    // And it stays aligned afterwards.
    let d = -1;
    act(() => {
      d = result.current.sendFrame(frame(), 2.5)!;
    });
    await flushSends();
    act(() => ws.reply(2));
    expect(onResult.mock.calls[2].slice(1)).toEqual([2.5, d]);
  });

  it("retires an errored frame without pairing it", async () => {
    const { result, ws, onResult } = await connectedFeed();
    act(() => {
      result.current.sendFrame(frame(), 0);
    });
    await flushSends();
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "error", frame_index: 0 }) });
    });
    expect(onResult).not.toHaveBeenCalled();
    expect(result.current.inFlightFrames).toBe(0);
    expect(result.current.frameErrors).toBe(1);
  });
});
