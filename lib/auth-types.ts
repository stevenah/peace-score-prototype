import "next-auth";
// Through next-auth, not "@auth/core/jwt": only next-auth is a declared
// dependency, and a clean pnpm install cannot resolve packages that are not.
import "next-auth/jwt";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      email: string;
      name?: string | null;
      role: string;
    };
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    id: string;
    role: string;
  }
}
