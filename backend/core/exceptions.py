class BookRAGException(Exception):
    """基础异常"""


class EmbeddingModelMismatch(BookRAGException):
    """KB 的 embedding 模型与当前配置不一致"""


class KBNotFoundException(BookRAGException):
    """知识库不存在"""


class DocumentNotFoundException(BookRAGException):
    """文档不存在"""


class SessionNotFoundException(BookRAGException):
    """会话不存在"""


class DocumentParseError(BookRAGException):
    """文档解析失败"""


class LLMProviderNotConfigured(BookRAGException):
    """LLM 提供商未配置"""


class EmbeddingProviderNotConfigured(BookRAGException):
    """Embedding 提供商未配置"""


class LLMEmptyResponse(BookRAGException):
    """模型没有产生任何正文内容（推理被截断或返回空）"""


class ProviderUnavailable(BookRAGException):
    """模型服务不可用（未启动 / 超时 / 鉴权失败 / 模型缺失），message 已面向用户"""


class EmbeddingProviderUnavailable(ProviderUnavailable):
    """Embedding 服务不可用"""


class LLMProviderUnavailable(ProviderUnavailable):
    """LLM 服务不可用"""
