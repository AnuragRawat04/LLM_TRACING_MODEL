from langfuse import get_client
import os

LF_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
LF_SECRET = os.getenv("LANGFUSE_SECRET_KEY")
LF_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
langfuse=get_client()