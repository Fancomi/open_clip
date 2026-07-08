"""dspy 模型接线: student=本地 vLLM(gemma/qwen), teacher=Opus(厂内代理)。

student 坑位同 prompt_lab: OpenAI 兼容端点 + 关思考模式。
teacher 从 ~/.claude/settings.json 读厂内代理配置 (非公网 anthropic)。
"""
import json
import os

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


def _parse_custom_headers(raw):
    """'comate_custom_header:{...json...}' -> {'comate_custom_header': '{...json...}'}"""
    if not raw or ":" not in raw:
        return {}
    name, val = raw.split(":", 1)
    return {name.strip(): val.strip()}


def make_teacher(*, temperature=1.0, max_tokens=4096, **kw):
    """Opus teacher, 走 ~/.claude/settings.json 厂内代理 (anthropic messages 协议)。
    GEPA reflection + 保真裁判共用。"""
    env = json.load(open(_SETTINGS)).get("env", {})
    base = env["ANTHROPIC_BASE_URL"].strip().rstrip("/")   # 不带 /v1, anthropic provider 自动补 /v1/messages
    token = env["ANTHROPIC_AUTH_TOKEN"].strip()
    model = env.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "Opus 4.8").strip()
    headers = _parse_custom_headers(env.get("ANTHROPIC_CUSTOM_HEADERS", ""))
    return dspy.LM(
        f"anthropic/{model}",
        api_base=base, api_key=token,
        temperature=temperature, max_tokens=max_tokens,
        extra_headers=headers or None, **kw,
    )
