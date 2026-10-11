const { Pool } = require("pg");
const bcrypt = require("bcryptjs");

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

  const pool = new Pool({ connectionString: process.env.DATABASE_URL });

  try {
    const hashedPassword = await bcrypt.hash(password, 12);

    const created = await pool.query(
      'INSERT INTO "User" (id, email, name, "hashedPassword", role, "createdAt", "updatedAt") VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW(), NOW()) ON CONFLICT (email) DO NOTHING',
      [email, "Admin", hashedPassword, "ADMIN"]
    );
    console.log(
      created.rowCount ? "Seed: created admin user." : "Seed: admin user already exists."
    );

    // An admin seeded by the old script still has the public password unless
    // someone changed it. Replace it; leave any other password alone.
    const candidates = await pool.query(
      'SELECT id, "hashedPassword" FROM "User" WHERE email = ANY($1)',
      [[email, LEGACY_EMAIL]]
    );
    for (const user of candidates.rows) {
      if (
        user.hashedPassword &&
        (await bcrypt.compare(LEGACY_PASSWORD, user.hashedPassword))
      ) {
        await pool.query(
          'UPDATE "User" SET "hashedPassword" = $1, "updatedAt" = NOW() WHERE id = $2',
          [hashedPassword, user.id]
        );
        console.log("Seed: replaced a default admin password.");
      }
    }
  } finally {
    await pool.end();
  }
}

main().catch((e) => {
  console.error("Seed failed:", e);
  process.exit(1);
});
