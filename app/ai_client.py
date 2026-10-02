import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

_client: Optional[httpx.AsyncClient] = None


def get_async_client() -> httpx.AsyncClient:
    """Returns a pooled, reusable HTTPX AsyncClient for high performance."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
            limits=httpx.Limits(
                max_keepalive_connections=20, max_connections=50
            ),
        )
    return _client


async def close_async_client() -> None:
    """Closes the pooled HTTPX AsyncClient on application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


class AIClient:
    """OpenAI & Anthropic compatible client with Bring-Your-Own-Key (BYOK) support.

    Zero server-side credentials: All URLs, keys, and model names are passed
    dynamically per request.
    """

    @staticmethod
    def _is_anthropic(base_url: str, api_key: Optional[str] = None) -> bool:
        url = (base_url or "").lower()
        if "anthropic.com" in url:
            return True
        if api_key and api_key.strip().startswith("sk-ant-"):
            return True
        return False

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Ensures the base_url does not have trailing slashes and properly points
        to the API root.
        """
        url = base_url.strip().rstrip("/")
        if "generativelanguage.googleapis.com" in url:
            if not url.endswith("/openai"):
                url = f"{url}/openai" if url.endswith("v1beta") else f"{url}/v1beta/openai"
            return url
        if "anthropic.com" in url:
            if not url.endswith("/v1"):
                url = f"{url}/v1"
            return url
        if not re.search(r"/v\d+$", url) and not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    @classmethod
    def _build_headers(cls, base_url: str, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if cls._is_anthropic(base_url, api_key):
            if api_key and api_key.strip():
                headers["x-api-key"] = api_key.strip()
            headers["anthropic-version"] = "2023-06-01"
        else:
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
        return headers

    @classmethod
    async def test_connection(
        cls, base_url: str, api_key: Optional[str], chat_model: str
    ) -> Dict[str, Any]:
        """Tests whether the provided Base URL, API Key, and Model can communicate."""
        normalized_url = cls._normalize_base_url(base_url)
        is_ant = cls._is_anthropic(base_url, api_key)
        client = get_async_client()

        if is_ant:
            endpoint = f"{normalized_url}/messages"
            headers = cls._build_headers(base_url, api_key)
            payload = {
                "model": chat_model,
                "messages": [{"role": "user", "content": "Hello! Reply with 'OK'."}],
                "max_tokens": 10,
                "temperature": 0.0,
            }
        else:
            endpoint = f"{normalized_url}/chat/completions"
            headers = cls._build_headers(base_url, api_key)
            payload = {
                "model": chat_model,
                "messages": [{"role": "user", "content": "Hello! Reply with 'OK'."}],
                "max_tokens": 10,
                "temperature": 0.0,
            }

        try:
            response = await client.post(
                endpoint, json=payload, headers=headers, timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                if is_ant:
                    contents = data.get("content", [])
                    content = contents[0].get("text", "") if contents else "OK"
                else:
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                return {
                    "success": True,
                    "message": f"連線成功！模型回應：{content}",
                }
            else:
                return {
                    "success": False,
                    "message": f"API 回應錯誤 (HTTP {response.status_code}): {response.text}",
                }
        except httpx.ConnectError:
            return {
                "success": False,
                "message": f"無法連線至 {normalized_url}，請檢查網址與網路設定（若是 Docker 訪問本機，請將 localhost 改為 host.docker.internal）。",
            }
        except httpx.TimeoutException:
            return {
                "success": False,
                "message": "連線逾時，目標伺服器未在 15 秒內回應。",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"連線發生未預期錯誤：{str(e)}",
            }

    @classmethod
    async def create_embeddings(
        cls,
        base_url: str,
        api_key: Optional[str],
        embedding_model: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generates embedding vectors for a list of texts using an OpenAI-compatible embedding endpoint."""
        if not texts:
            return []

        # If user accidentally sets an Anthropic endpoint for embeddings, guide them
        if "anthropic.com" in base_url.lower():
            raise RuntimeError(
                "Anthropic 官方 API 本身不提供文本向量嵌入模型 (Embedding)。\n"
                "請在「嵌入模型」中使用 Google Gemini (text-embedding-004)、OpenAI (text-embedding-3-small) 或本地 Ollama (nomic-embed-text)。"
            )

        normalized_url = cls._normalize_base_url(base_url)
        endpoint = f"{normalized_url}/embeddings"
        headers = cls._build_headers(base_url, api_key)

        all_embeddings: List[List[float]] = []
        client = get_async_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": embedding_model,
                "input": batch,
            }
            response = await client.post(
                endpoint, json=payload, headers=headers, timeout=60.0
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding API 呼叫失敗 (HTTP {response.status_code}): {response.text}"
                )

            data = response.json()
            items = data.get("data", [])
            items.sort(key=lambda x: x.get("index", 0))
            for item in items:
                all_embeddings.append(item["embedding"])

        return all_embeddings

    @classmethod
    async def chat_completion(
        cls,
        base_url: str,
        api_key: Optional[str],
        chat_model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Calls the Chat Completions endpoint (non-streaming)."""
        normalized_url = cls._normalize_base_url(base_url)
        is_ant = cls._is_anthropic(base_url, api_key)
        client = get_async_client()

        if is_ant:
            endpoint = f"{normalized_url}/messages"
            headers = cls._build_headers(base_url, api_key)
            system_prompt = ""
            filtered_msgs = []
            for m in messages:
                if m.get("role") == "system":
                    system_prompt += m.get("content", "") + "\n"
                else:
                    filtered_msgs.append({"role": m.get("role"), "content": m.get("content", "")})

            payload = {
                "model": chat_model,
                "messages": filtered_msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                payload["system"] = system_prompt.strip()

            response = await client.post(
                endpoint, json=payload, headers=headers, timeout=120.0
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Claude API 呼叫失敗 (HTTP {response.status_code}): {response.text}"
                )
            data = response.json()
            contents = data.get("content", [])
            return contents[0].get("text", "").strip() if contents else ""

        else:
            endpoint = f"{normalized_url}/chat/completions"
            headers = cls._build_headers(base_url, api_key)
            payload = {
                "model": chat_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = await client.post(
                endpoint, json=payload, headers=headers, timeout=120.0
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Chat API 呼叫失敗 (HTTP {response.status_code}): {response.text}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("Chat API 未回傳任何候選回應 (choices 為空)")

            return choices[0].get("message", {}).get("content", "").strip()

    @classmethod
    async def chat_completion_stream(
        cls,
        base_url: str,
        api_key: Optional[str],
        chat_model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Streams token deltas from Chat Completions."""
        normalized_url = cls._normalize_base_url(base_url)
        is_ant = cls._is_anthropic(base_url, api_key)
        client = get_async_client()

        if is_ant:
            endpoint = f"{normalized_url}/messages"
            headers = cls._build_headers(base_url, api_key)
            system_prompt = ""
            filtered_msgs = []
            for m in messages:
                if m.get("role") == "system":
                    system_prompt += m.get("content", "") + "\n"
                else:
                    filtered_msgs.append({"role": m.get("role"), "content": m.get("content", "")})

            payload = {
                "model": chat_model,
                "messages": filtered_msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if system_prompt:
                payload["system"] = system_prompt.strip()

            async with client.stream(
                "POST", endpoint, json=payload, headers=headers, timeout=120.0
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(
                        f"Claude API 串流呼叫失敗 (HTTP {response.status_code}):"
                        f" {err_text.decode('utf-8', errors='ignore')}"
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type")
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

        else:
            endpoint = f"{normalized_url}/chat/completions"
            headers = cls._build_headers(base_url, api_key)
            payload = {
                "model": chat_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            async with client.stream(
                "POST", endpoint, json=payload, headers=headers, timeout=120.0
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(
                        f"Chat API 串流呼叫失敗 (HTTP {response.status_code}):"
                        f" {err_text.decode('utf-8', errors='ignore')}"
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
