import { z } from "zod/v4";

const envSchema = z.object({
  DATABASE_URL: z.string().min(1),
  NEXTAUTH_SECRET: z.string().min(16, "NEXTAUTH_SECRET must be at least 16 characters"),
  NEXTAUTH_URL: z.url().optional(),
  AWS_REGION: z.string().min(1).optional(),
  AWS_ACCESS_KEY_ID: z.string().min(1).optional(),
  AWS_SECRET_ACCESS_KEY: z.string().min(1).optional(),
  S3_BUCKET_NAME: z.string().min(1).optional(),
  ML_BACKEND_URL: z.string().default("http://localhost:8000"),
  ML_BACKEND_PUBLIC_URL: z.string().optional(),
});

function validateEnv() {
  const result = envSchema.safeParse(process.env);
  if (!result.success) {
    const formatted = z.prettifyError(result.error);
    console.error("Invalid environment variables:\n", formatted);
    throw new Error("Invalid environment variables. Check server logs.");
  }
  return result.data;
}

export const env = validateEnv();
