const { Pool } = require("pg");
const bcrypt = require("bcryptjs");

async function main() {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });

  try {
    const existing = await pool.query(
      'SELECT id FROM "User" WHERE email = $1',
      ["admin@peace.dev"]
    );

    if (existing.rows.length > 0) {
      console.log("Seed: admin user already exists, skipping.");
      return;
    }

    const hashedPassword = await bcrypt.hash("admin123", 12);
    await pool.query(
      'INSERT INTO "User" (id, email, name, "hashedPassword", role, "createdAt", "updatedAt") VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW(), NOW())',
      ["admin@peace.dev", "Admin", hashedPassword, "ADMIN"]
    );

    console.log("Seed: created admin user.");
  } finally {
    await pool.end();
  }
}

main().catch((e) => {
  console.error("Seed failed:", e);
  process.exit(1);
});
