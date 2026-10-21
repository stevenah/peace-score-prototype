import "dotenv/config";
import { PrismaLibSql } from "@prisma/adapter-libsql";
import { PrismaClient } from "../generated/prisma/client";
import bcrypt from "bcryptjs";

// Local development only. Production seeds through prisma/seed-deploy.ts,
// which refuses this password.
const adapter = new PrismaLibSql({ url: process.env.DATABASE_URL! });
const prisma = new PrismaClient({ adapter });

async function main() {
  const hashedPassword = await bcrypt.hash("admin123", 12);

  const user = await prisma.user.upsert({
    where: { email: "admin@peace.dev" },
    update: {},
    create: {
      email: "admin@peace.dev",
      name: "Admin",
      hashedPassword,
      role: "ADMIN",
    },
  });

  console.log(`Seeded user: ${user.email}`);
}

main()
  .then(() => prisma.$disconnect())
  .catch((e) => {
    console.error(e);
    prisma.$disconnect();
    process.exit(1);
  });
