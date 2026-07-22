import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 30_000,
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      '请求失败'
    return Promise.reject(new Error(message))
  },
)

export default client
