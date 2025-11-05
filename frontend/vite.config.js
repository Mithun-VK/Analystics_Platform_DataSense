// vite.config.ts
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import compression from 'vite-plugin-compression';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    // ✅ FIXED: Explicitly set root for development
    root: process.cwd(),

    plugins: [
      react({
        jsxRuntime: 'automatic',
        jsxImportSource: 'react',
      }),
      // ✅ Only run compression and visualization in build mode
      command === 'build' &&
        compression({
          algorithm: 'brotli',
          ext: '.br',
          threshold: 10240,
          verbose: true,
        }),
      command === 'build' &&
        compression({
          algorithm: 'gzip',
          ext: '.gz',
          threshold: 10240,
          verbose: true,
        }),
      command === 'build' &&
        visualizer({
          open: false,
          filename: 'dist/stats.html',
          gzipSize: true,
          brotliSize: true,
        }),
    ].filter(Boolean),

    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@pages': path.resolve(__dirname, './src/pages'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@services': path.resolve(__dirname, './src/services'),
        '@store': path.resolve(__dirname, './src/store'),
        '@types': path.resolve(__dirname, './src/types'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@styles': path.resolve(__dirname, './src/styles'),
        '@assets': path.resolve(__dirname, './src/assets'),
      },
    },

    server: {
      port: 5174,
      host: true,
      strictPort: false,
      open: false,
      // ✅ FIXED: Better HMR configuration
      hmr: {
        protocol: 'ws',
        host: 'localhost',
        port: 5174,
      },
      // ✅ FIXED: Correct API proxy URL
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '/api/v1'),
        },
      },
    },

    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: mode !== 'production',
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: mode === 'production',
          drop_debugger: mode === 'production',
        },
      },
      rollupOptions: {
        // ✅ FIXED: Explicitly include index.html
        input: path.resolve(__dirname, 'index.html'),
        output: {
          manualChunks: {
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],
            'vendor-query': ['@tanstack/react-query'],
            'vendor-utils': ['axios', 'zustand', 'lodash-es', 'date-fns'],
            'vendor-ui': ['recharts', 'lucide-react', 'sonner'],
          },
          chunkFileNames: 'assets/[name]-[hash].js',
          entryFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash][extname]',
        },
      },
      commonjsOptions: {
        transformMixedEsModules: true,
      },
      reportCompressedSize: true,
      chunkSizeWarningLimit: 1500,
    },

    preview: {
      port: 4173,
      host: true,
      strictPort: false,
      open: false,
    },

    define: {
      __APP_NAME__: JSON.stringify(env.VITE_APP_NAME || 'DataAnalytics'),
      __APP_VERSION__: JSON.stringify(env.VITE_APP_VERSION || '1.0.0'),
      __API_URL__: JSON.stringify(
        env.VITE_API_URL || 'http://localhost:8000/api/v1'
      ),
    },

    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        'zustand',
        '@tanstack/react-query',
        'axios',
        'recharts',
        'lucide-react',
      ],
    },
  };
});
