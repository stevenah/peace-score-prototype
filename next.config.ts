import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyClientMaxBodySize: "500mb",
  },
  async rewrites() {
    return [
      {
        source: "/ml-api/:path*",
        destination: `${process.env.ML_BACKEND_URL || "http://localhost:8000"}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
