import json
import re
from io import BytesIO
from typing import Any, Union
import PIL
import PIL.Image
import PIL.ImageOps
import base64
import os
import httpx2 as httpx
####################### CONSTANT #######################
TOOL_PROMPT = '''# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>'''

ROLES = ("system", "user", "assistant", "tool")

IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_TOKEN_LEN = 64  # siglip2-base-patch32-256: 256/32 = 8x8 = 64 个 patch

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S
)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

####################### tools #######################
def render_tools(tools) -> str:
    """渲染 tools 描述。tools 允许传 list，也允许传 JSON 字符串。"""
    if isinstance(tools, str):
        tools = json.loads(tools)
    if isinstance(tools, dict):
        raise TypeError("tools 应该传 list[dict]，收到单个 dict")
    if not tools:
        return ""
    body = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return TOOL_PROMPT.format(tools=body)

def load_image(
    image: Union[str, "PIL.Image.Image"],
    timeout: float | None = None,
) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(BytesIO(httpx.get(image, timeout=timeout, follow_redirects=True).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif not isinstance(image, PIL.Image.Image):
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def render_content(content) -> tuple[str, list]:
    """把 content 统一成字符串，并带回其中的图片地址。

    - str 原样返回，None 返回空串（assistant 只有 tool_calls 时常见）
    - parts 数组：text 取原文；image 展开成 IMAGE_TOKEN_LEN 个 <|image_pad|>，
      地址按出现顺序追加到返回的 images，保证和文本里的 pad 顺序一一对应
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        parts, images = [], []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError(f"content 分段必须是 dict，收到 {type(item).__name__}")
            if "image" in item or "image_url" in item or item.get("type") == "image":
                ref = item.get("image") or item.get("image_url") or item.get("url")
                if ref is None:
                    raise ValueError(f"image 分段缺少 image / image_url / url: {item!r}")
                if isinstance(ref, (list, tuple, dict)):
                    raise ValueError(
                        f"image 分段里的 image 应是单个地址（str / PIL.Image），"
                        f"收到的却是 {type(ref).__name__}: {ref!r}"
                    )
                ref = load_image(image=ref)
                images.append(ref)
                parts.append(IMAGE_PAD_TOKEN * IMAGE_TOKEN_LEN)
            elif "text" in item:
                parts.append(item["text"])
            else:
                raise ValueError(f"无法识别的 content 分段: {item!r}")
        return "".join(parts), images
    raise TypeError(f"不支持的 content 类型: {type(content).__name__}")


def format_tool_call(tool_call: dict) -> str:
    """把一次工具调用渲染成 <tool_call>...</tool_call>"""
    if "function" in tool_call and tool_call.get("type", "function") == "function":
        tool_call = tool_call["function"]
    name = tool_call["name"]
    params = ""
    for key, value in (tool_call.get("arguments") or {}).items():
        # 官方模板对 dict/list 用 tojson，其余直接 str
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        params += f"<parameter={key}>\n{value}\n</parameter>\n"
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def decode_tool_call(assistant_content: str) -> str:
    '''把 xml 的 tool call 转为 "{"name": "click", "arguments": {"x": 105, "y": 132}}"

    参数值优先按 JSON 解析（dict/list/数字/bool/null），解析失败就当字符串。
    <parameter> 里不存类型信息，所以 bool/None 被 format_tool_call 写成 True/None
    之后，回读只能是字符串（格式本身的限制）。
    '''
    if not assistant_content:
        raise ValueError("assistant 内容为空，没有可解析的 tool call")
    match = _TOOL_CALL_RE.search(assistant_content)
    if match is None:
        raise ValueError(f"没有找到 <tool_call>...</tool_call> 块: {assistant_content!r}")

    arguments = {}
    for param in _PARAM_RE.finditer(match.group(2)):
        key, raw = param.group(1), param.group(2).strip()
        try:
            arguments[key] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[key] = raw
    return json.dumps({"name": match.group(1), "arguments": arguments}, ensure_ascii=False)


def chat_markers(tokenizer=None) -> tuple[str, str]:
    """对话轮次开始 / 结束标记，返回 ``(bos_token, eos_token)``。

    优先用 tokenizer 的 special token，缺省回落到 Qwen 的 chatml 标记。
    Qwen 系 tokenizer 的 bos_token 是 None（只有 eos_token=<|im_end|>）。
    """
    bos = getattr(tokenizer, "bos_token", None) or "<|im_start|>"
    eos = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    return bos, eos

def apply_chat_template(
    messages: list,
    tools=None,
    add_generation_prompt: bool = False,
    tokenizer=None
) -> dict:
    '''
    数据格式
    "messages": [{"role": "", "content": ""}],
    "tools": []

    处理方式（目前只支持 image + text -> text）
    content 如果有多模态统一用 parts 数组，key 为 image 时展开成
    image_pad_token * IMAGE_TOKEN_LEN，图片地址按出现顺序放进 pixel_vals。

    返回
    {
        "inputs": "处理后的字符串，用于直接 tokenizer.encode",
        "pixel_vals": ["PIL.Image.Image", ...]  # 图片数据数组，交给视觉塔读
    }
    '''
    bos_token, eos_token = chat_markers(tokenizer)

    if not messages:
        raise ValueError("messages 为空，至少要有一条消息")
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in ROLES:
            raise ValueError(f"第 {idx} 条消息的 role 不支持: {role!r}")
        if role == "system" and idx != 0:
            raise ValueError("system 消息必须放在第一位")

    tools_text = render_tools(tools)

    system_content = ""
    start = 0
    if messages[0]["role"] == "system":
        system_content, _ = render_content(messages[0].get("content"))
        system_content = system_content.strip()
        start = 1

    chat = ""
    if tools_text:
        chat += f"{bos_token}system\n{tools_text}"
        if system_content:
            chat += f"\n\n{system_content}"
        chat += f"{eos_token}\n"
    elif start == 1:  # 首条是 system 但内容为空，官方也会输出一个空的 system 块
        chat += f"{bos_token}system\n{system_content}{eos_token}\n"

    # if not any(msg["role"] == "user" for msg in messages):
    #     raise ValueError("没有 user query，样本不合法")

    images = []  # 全局图片地址数组，顺序 == 文本里 <|image_pad|> 的顺序
    for msg in messages[start:]:
        role = msg["role"]
        content, msg_images = render_content(msg.get("content"))
        content = content.strip()
        images.extend(msg_images)

        if role == "user":
            chat += f"{bos_token}user\n{content}{eos_token}\n"
        elif role == "tool":  # 本项目约定：tool 结果单独一轮
            chat += f"{bos_token}tool\n{content}{eos_token}\n"
        elif role == "assistant":
            chat += f"{bos_token}assistant\n{content}"
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                if content:
                    chat += "\n\n"
                chat += "\n".join(format_tool_call(call) for call in tool_calls)
            chat += f"{eos_token}\n"

    if add_generation_prompt:
        chat += f"{bos_token}assistant\n"

    return {"inputs": chat, "pixel_vals": images}
