/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端配置 API_TOKEN 时，构建期注入同一个值 */
  readonly VITE_API_TOKEN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
