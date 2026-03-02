/** @type {import('next').NextConfig} */
const nextConfig = {
  // "standalone" is for Docker/self-hosted; Vercel handles this automatically
  ...(process.env.VERCEL ? {} : { output: "standalone" }),
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

module.exports = nextConfig;
