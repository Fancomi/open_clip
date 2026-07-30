"""dspy 模型接线: student=本地 vLLM(gemma/qwen), teacher=Opus(厂内代理)。

student 坑位同 prompt_lab: OpenAI 兼容端点 + 关思考模式。
teacher 从 ~/.claude/settings.json 读厂内代理配置 (非公网 anthropic)。
"""
import json
import os
import re

import dspy

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # 本地回环 + 厂内代理都不走系统代理

MODELS = {
    "qwen":  ("/dev/shm/models/Qwen3.6-35B-A3B-FP8", [8005, 8006, 8007, 8008]),
    "gemma": ("/dev/shm/models/gemma-4-26B-A4B-it",  [8001, 8002, 8003, 8004]),
}
_SETTINGS = os.path.expanduser("~/.claude/settings.json")


def make_student(which="qwen", *, port=None, think=False, temperature=0.3,
                 max_tokens=1024, **kw):
    """本地 vLLM student。think=False 关思考模式 (dspy 必需)。"""
    model_id, ports = MODELS[which]
    return dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://127.0.0.1:{port or ports[0]}/v1",
        api_key="EMPTY", temperature=temperature, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        **kw,
    )


_HDR_RE = re.compile(
    r"([A-Za-z0-9_-]+)\s*:\s*(\{.*?\}|[^,\n]+?)"
    r"(?=\s*(?:,\s*[A-Za-z0-9_-]+\s*:|\n|$))"
)


def _parse_custom_headers(raw):
    """'comate_custom_header:{...json...}' -> {'comate_custom_header': '{...json...}'}

    格式是 HTTP 头列表 '名:值', 可用逗号或换行分隔多个头, comate_custom_header 的值
    是 JSON (可能带 \\" 转义)。oneapi 拿里面的 source/username/agentId 做归属统计,
    且对残头一律返回 200 (实测: 截断 JSON / 缺 source / 无头都不报错), 所以只能在这
    边保证: JSON 必须能解析且 source 非空, 否则宁可抛错也不发不完整的归属信息。
    """
    if not raw:
        return {}
    headers = {m.group(1).strip(): m.group(2).strip() for m in _HDR_RE.finditer(raw)}
    val = headers.get("comate_custom_header")
    if val is None:
        if "comate_custom_header" in raw:   # 写了但没解析出来 = JSON 残缺, 不能静默丢
            raise RuntimeError(f"comate_custom_header 解析失败 (JSON 可能被截断): {raw!r}")
        return headers
    try:
        meta = json.loads(val.replace('\\"', '"'))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"comate_custom_header 不是合法 JSON, 无法保证 source 正确: {val!r}") from e
    if not meta.get("source"):
        raise RuntimeError(f"comate_custom_header 缺少 source 字段: {meta!r}")
    headers["comate_custom_header"] = json.dumps(meta, ensure_ascii=False)  # 规范化, 去掉转义残留
    return headers


def make_teacher(*, temperature=1.0, max_tokens=4096, **kw):
    """gpt-5.6-sol teacher, 走 ~/.claude/settings.json 厂内代理 (OpenAI chat 协议)。
    GEPA reflection + 保真裁判共用。key/自定义头复用 settings, 仅换模型与协议。

    注: 之前用 Opus 4.8 (anthropic messages 协议, base 不带 /v1); gpt-5.6-sol
    是 OpenAI chat 协议, 前缀 openai/ 且 base 需补 /v1。
    """
    env = json.load(open(_SETTINGS)).get("env", {})
    base = env["ANTHROPIC_BASE_URL"].strip().rstrip("/") + "/v1"   # openai provider 需 /v1
    token = env["ANTHROPIC_AUTH_TOKEN"].strip()
    headers = _parse_custom_headers(env.get("ANTHROPIC_CUSTOM_HEADERS", ""))
    if "comate_custom_header" not in headers:   # 缺归属头 -> oneapi 侧统计不到, 直接拦
        raise RuntimeError(f"{_SETTINGS} 缺少 ANTHROPIC_CUSTOM_HEADERS 的 comate_custom_header")
    return dspy.LM(
        "openai/gpt-5.6-sol",
        api_base=base, api_key=token,
        temperature=temperature, max_tokens=max_tokens,
        extra_headers=headers, **kw,
    )
