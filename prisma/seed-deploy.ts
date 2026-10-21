import "dotenv/config";
import { PrismaLibSql } from "@prisma/adapter-libsql";
import bcrypt from "bcryptjs";
import { PrismaClient } from "../generated/prisma/client";

// Runs at every boot of the production machine (Dockerfile.frontend CMD),
// after the migrations. Creates the admin account once, from secrets.
//
// This script used to create the admin with these hardcoded values, and the
// repository is public. No account may keep that password.
const LEGACY_EMAIL = "admin@peace.dev";
const LEGACY_PASSWORD = "admin123";

async function main() {
  const email = process.env.SEED_ADMIN_EMAIL || LEGACY_EMAIL;
  const password = process.env.SEED_ADMIN_PASSWORD;

  if (!password) {
    console.log("Seed: SEED_ADMIN_PASSWORD is not set, skipping.");
    return;
  }
  if (password === LEGACY_PASSWORD) {
    throw new Error("SEED_ADMIN_PASSWORD is the old public default; choose another.");
  }

  const adapter = new PrismaLibSql({ url: process.env.DATABASE_URL! });
  const prisma = new PrismaClient({ adapter });

  try {
    const hashedPassword = await bcrypt.hash(password, 12);

    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing) {
      console.log("Seed: admin user already exists.");
    } else {
      await prisma.user.create({
        data: { email, name: "Admin", hashedPassword, role: "ADMIN" },
      });
      console.log("Seed: created admin user.");
    }

    // An admin seeded by the old script still has the public password unless
    // someone changed it. Replace it; leave any other password alone.
    const candidates = await prisma.user.findMany({
      where: { email: { in: [email, LEGACY_EMAIL] } },
      select: { id: true, hashedPassword: true },
    });
    for (const user of candidates) {
      if (user.hashedPassword && (await bcrypt.compare(LEGACY_PASSWORD, user.hashedPassword))) {
        await prisma.user.update({ where: { id: user.id }, data: { hashedPassword } });
        console.log("Seed: replaced a default admin password.");
      }
    }
  } finally {
    await prisma.$disconnect();
  }
}

main().catch((e) => {
  console.error("Seed failed:", e);
  process.exit(1);
});
