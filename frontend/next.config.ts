import type { NextConfig } from "next";

// Next.js 16 + Turbopack tries to auto-detect the workspace root by walking
// up until it finds a lockfile. On the dev machine the walk finds a parent
// `pnpm-lock.yaml`; on Vercel's deploy container there is no parent
// lockfile and the auto-detection returns `undefined`, which makes the
// "Applying modifyConfig from Vercel" step crash with:
//   TypeError: The "path" argument must be of type string. Received undefined.
// Pinning both `turbopack.root` and `outputFileTracingRoot` to the
// frontend directory removes the ambiguity in both environments.
const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
  },
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
