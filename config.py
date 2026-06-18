"""
集中配置 — 金融LLM数据建设与评测平台
"""
import os

# ─── 路径 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval_results")

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ─── 加载 Secrets ───
def _load_secrets():
    """尝试从 .streamlit/secrets.toml 加载配置（兼容非Streamlit环境）"""
    secrets = {}
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return secrets
        try:
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            for k, v in data.items():
                secrets[k] = str(v).strip('"\'')
        except:
            pass
    return secrets

_secrets = _load_secrets()

# ─── Embedding ───
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# HF端点：优先用环境变量/Secrets，默认用镜像
_hf = os.getenv("HF_ENDPOINT") or _secrets.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _hf

# ─── LLM ───
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY") or _secrets.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("API_BASE_URL") or _secrets.get("API_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL") or _secrets.get("LLM_MODEL", "deepseek-chat")
