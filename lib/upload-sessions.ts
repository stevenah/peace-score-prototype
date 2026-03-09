import { createWriteStream, mkdirSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { WriteStream } from "fs";

interface UploadSession {
  userId: string;
  userEmail: string | null;
  filePath: string;
  stream: WriteStream;
  chunksReceived: number;
  totalChunks: number;
  createdAt: number;
}

// Store sessions on globalThis so they survive Next.js module re-evaluation
// (hot reload in dev, route handler re-imports, etc.)
const GLOBAL_KEY = "__peace_upload_sessions" as const;
const g = globalThis as typeof globalThis & {
  [GLOBAL_KEY]?: Map<string, UploadSession>;
  __peace_upload_cleanup?: ReturnType<typeof setInterval>;
};

if (!g[GLOBAL_KEY]) {
  g[GLOBAL_KEY] = new Map();
}
const sessions = g[GLOBAL_KEY];

// Clean up stale sessions every 10 minutes (only one interval)
const STALE_TIMEOUT = 30 * 60 * 1000; // 30 minutes

if (!g.__peace_upload_cleanup) {
  g.__peace_upload_cleanup = setInterval(() => {
    const now = Date.now();
    for (const [id, session] of sessions) {
      if (now - session.createdAt > STALE_TIMEOUT) {
        cleanupSession(id, session);
      }
    }
  }, 10 * 60 * 1000);
}

const UPLOAD_DIR = join(tmpdir(), "peace-chunked-uploads");
mkdirSync(UPLOAD_DIR, { recursive: true });

export function registerUpload(userId: string, userEmail: string | null): string {
  const uploadId = `up_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const filePath = join(UPLOAD_DIR, `${uploadId}.bin`);
  const stream = createWriteStream(filePath, { flags: "w" });

  sessions.set(uploadId, {
    userId,
    userEmail,
    filePath,
    stream,
    chunksReceived: 0,
    totalChunks: 0,
    createdAt: Date.now(),
  });

  return uploadId;
}

export function getSession(uploadId: string): UploadSession | undefined {
  return sessions.get(uploadId);
}

export function appendChunk(
  uploadId: string,
  chunk: Buffer,
  chunkIndex: number,
  totalChunks: number,
): { complete: boolean } {
  const session = sessions.get(uploadId);
  if (!session) throw new Error("Upload session not found");

  if (session.totalChunks === 0) {
    session.totalChunks = totalChunks;
  }

  // Write chunk — chunks arrive sequentially from the client
  session.stream.write(chunk);
  session.chunksReceived++;

  if (session.chunksReceived >= session.totalChunks) {
    session.stream.end();
    return { complete: true };
  }

  return { complete: false };
}

export function getFilePath(uploadId: string): string | undefined {
  return sessions.get(uploadId)?.filePath;
}

export function waitForFlush(uploadId: string): Promise<void> {
  const session = sessions.get(uploadId);
  if (!session) return Promise.resolve();

  return new Promise((resolve, reject) => {
    if (session.stream.writableFinished) {
      resolve();
    } else {
      session.stream.on("finish", resolve);
      session.stream.on("error", reject);
    }
  });
}

export function removeSession(uploadId: string) {
  const session = sessions.get(uploadId);
  if (session) {
    cleanupSession(uploadId, session);
  }
}

function cleanupSession(id: string, session: UploadSession) {
  try {
    session.stream.destroy();
  } catch {}
  try {
    unlinkSync(session.filePath);
  } catch {}
  sessions.delete(id);
}
