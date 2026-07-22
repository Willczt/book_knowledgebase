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
