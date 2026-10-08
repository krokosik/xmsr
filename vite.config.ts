import { defineConfig } from "vitest/config";

export default defineConfig(({ command }) => ({
  define: command === "build" ? { "process.env.NODE_ENV": JSON.stringify("production") } : {},
  build: {
    outDir: "src/xmsr/static",
    lib: {
      entry: "frontend/widget.tsx",
      formats: ["es"],
      fileName: () => "runner.js",
      cssFileName: "runner",
    },
  },
  test: {
    environment: "jsdom",
    include: ["frontend/**/*.test.tsx"],
    setupFiles: ["frontend/test-setup.ts"],
  },
}));
