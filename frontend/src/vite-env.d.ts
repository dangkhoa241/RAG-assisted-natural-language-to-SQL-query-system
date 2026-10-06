/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** The API's origin in production, e.g. https://nl2sql-api.onrender.com (no trailing slash). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
