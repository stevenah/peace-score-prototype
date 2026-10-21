import { PrismaLibSql } from "@prisma/adapter-libsql";
import { PrismaClient } from "@/generated/prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

/**
 * One SQLite file, named by DATABASE_URL: `file:./dev.db` locally,
 * `file:/data/peace.db` on the Fly volume. WAL mode lets reads proceed while a
 * write is in flight, and the busy timeout makes a second writer wait instead
 * of failing; both matter because the API routes run concurrently.
 */
function createPrismaClient() {
  const adapter = new PrismaLibSql({ url: process.env.DATABASE_URL! });
  const client = new PrismaClient({ adapter });
  void client
    .$queryRawUnsafe("PRAGMA journal_mode = WAL")
    .then(() => client.$queryRawUnsafe("PRAGMA busy_timeout = 5000"))
    .catch((err: unknown) => console.error("SQLite pragmas failed:", err));
  return client;
}

export const prisma = globalForPrisma.prisma ?? createPrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
