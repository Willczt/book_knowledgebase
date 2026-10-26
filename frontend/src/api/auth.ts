// 可选 API Token：后端配置 API_TOKEN 时，前端需在构建期注入 VITE_API_TOKEN。
// 未配置时返回空对象，本地单机使用不受影响。
const token = (import.meta.env.VITE_API_TOKEN as string | undefined)?.trim()

export const authHeaders: Record<string, string> = token
  ? { Authorization: `Bearer ${token}` }
  : {}
