import NextAuth from "next-auth";
import { authConfig } from "@/lib/auth.config";
import { NextResponse } from "next/server";

const { auth } = NextAuth(authConfig);

export default auth((req) => {
  const { pathname } = req.nextUrl;
  const isLoggedIn = !!req.auth;

  const publicRoutes = ["/login", "/register"];
  const isPublicRoute = publicRoutes.some((route) =>
    pathname.startsWith(route),
  );
  const isAuthApi = pathname.startsWith("/api/auth");
  const isOtherApi = pathname.startsWith("/api/");

  // Allow auth API routes
  if (isAuthApi) {
    return NextResponse.next();
  }

  // Redirect logged-in users away from login/register
  if (isLoggedIn && isPublicRoute) {
    return NextResponse.redirect(new URL("/dashboard", req.nextUrl));
  }

  // Allow public routes
  if (isPublicRoute) {
    return NextResponse.next();
  }

  // Allow other API routes (upload, analysis polling)
  if (isOtherApi) {
    return NextResponse.next();
  }

  // Redirect unauthenticated users to login
  if (!isLoggedIn) {
    return NextResponse.redirect(new URL("/login", req.nextUrl));
  }

  return NextResponse.next();
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|public/).*)"],
};
