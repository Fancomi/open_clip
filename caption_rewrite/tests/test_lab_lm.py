"""_parse_custom_headers: oneapi 归属头必须完整 (source 非空)。

lab_lm 只在构造 LM 时用 dspy, 无 dspy 环境下临时塞个 stub 把它导进来, 导完立刻从
sys.modules 摘掉, 免得影响同批其他测试模块的真实 import 行为。
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest

_stubbed = False
try:
    import dspy  # noqa: F401
except ImportError:
    sys.modules["dspy"] = types.ModuleType("dspy")
    _stubbed = True
try:
    import lab_lm
finally:
    if _stubbed:
        sys.modules.pop("dspy", None)

parse = lab_lm._parse_custom_headers


def test_single_header_json_kept():
    raw = 'comate_custom_header:{"agentId": "ducc:user:x", "username": "x", "repo": "", "source": "ducc"}'
    got = parse(raw)
    assert json.loads(got["comate_custom_header"])["source"] == "ducc"


def test_escaped_quotes_unescaped():
    raw = 'comate_custom_header:{\\"agentId\\": \\"ducc:user:x\\", \\"source\\": \\"ducc\\"}'
    meta = json.loads(parse(raw)["comate_custom_header"])   # 带 \" 也要能解析
    assert meta == {"agentId": "ducc:user:x", "source": "ducc"}


def test_multiple_headers_split_on_comma():
    raw = 'comate_custom_header:{"source": "ducc", "username": "x"},x-trace:abc'
    got = parse(raw)
    assert got["x-trace"] == "abc"                          # 逗号后的头不被吞进 JSON
    assert json.loads(got["comate_custom_header"])["source"] == "ducc"


def test_agent_id_colon_not_split():
    """值里的 ':' (ducc:user:x) 不能把头名切错。"""
    raw = 'comate_custom_header:{"agentId": "ducc:user:x", "source": "ducc"}'
    assert list(parse(raw)) == ["comate_custom_header"]


def test_nested_json_and_comma_in_value():
    """嵌套对象 / 值里的逗号都不能把 JSON 截断。"""
    got = parse('comate_custom_header:{"source":"ducc","ext":{"a":1},"note":"a,b"},x-t:1')
    meta = json.loads(got["comate_custom_header"])
    assert meta == {"source": "ducc", "ext": {"a": 1}, "note": "a,b"}
    assert got["x-t"] == "1"


def test_empty_raw():
    assert parse("") == {}
    assert parse(None) == {}


@pytest.mark.parametrize("raw", [
    'comate_custom_header:{"agentId": "a", "source": ""}',      # source 空
    'comate_custom_header:{"agentId": "a"}',                    # 无 source
    'comate_custom_header:{"agentId": "a", "repo": ""',         # JSON 截断
    'comate_custom_header:{"agentId": "a", "source": "ducc"',   # 截断但含 source
])
def test_incomplete_source_raises(raw):
    """oneapi 对残头照样返回 200, 归属信息会静默丢失, 所以这里必须硬失败。"""
    with pytest.raises(RuntimeError, match="comate_custom_header"):
        parse(raw)


def test_real_settings_header_ok():
    """本机 ~/.claude/settings.json 里的头必须合规 (缺失则跳过)。"""
    path = os.path.expanduser("~/.claude/settings.json")
    if not os.path.exists(path):
        pytest.skip("no settings.json")
    raw = json.load(open(path)).get("env", {}).get("ANTHROPIC_CUSTOM_HEADERS", "")
    if not raw:
        pytest.skip("no ANTHROPIC_CUSTOM_HEADERS")
    assert json.loads(parse(raw)["comate_custom_header"])["source"]


def _fake_settings(tmp_path, env):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"env": env}))
    return str(p)


_BASE_ENV = {"ANTHROPIC_BASE_URL": "https://oneapi-comate.baidu-int.com/",
             "ANTHROPIC_AUTH_TOKEN": " tok "}


def test_make_teacher_passes_header(tmp_path, monkeypatch):
    env = dict(_BASE_ENV, ANTHROPIC_CUSTOM_HEADERS=
               'comate_custom_header:{"agentId": "ducc:user:x", "source": "ducc"}')
    monkeypatch.setattr(lab_lm, "_SETTINGS", _fake_settings(tmp_path, env))
    monkeypatch.setattr(lab_lm.dspy, "LM", lambda *a, **kw: (a, kw), raising=False)
    _, kw = lab_lm.make_teacher()
    assert kw["api_base"] == "https://oneapi-comate.baidu-int.com/v1"
    assert kw["api_key"] == "tok"
    assert json.loads(kw["extra_headers"]["comate_custom_header"])["source"] == "ducc"


def test_make_teacher_rejects_missing_header(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_lm, "_SETTINGS", _fake_settings(tmp_path, dict(_BASE_ENV)))
    monkeypatch.setattr(lab_lm.dspy, "LM", lambda *a, **kw: (a, kw), raising=False)
    with pytest.raises(RuntimeError, match="comate_custom_header"):
        lab_lm.make_teacher()
