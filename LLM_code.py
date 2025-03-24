import os
from openai import OpenAI
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
import arxiv
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis

# 初始化DashScope客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def llm(query, history=[], user_stop_words=[]):
    """
    使用DashScope API调用大模型
    """
    try:
        messages = [{'role': 'system', 'content': '你是雷南方制造的智能助手'}]

        # 构建对话历史
        for hist in history:
            messages.append({'role': 'user', 'content': hist[0]})
            messages.append({'role': 'assistant', 'content': hist[1]})
        messages.append({'role': 'user', 'content': query})

        completion = client.chat.completions.create(
            model="qwen-max-latest",  # 默认使用qwen-plus模型
            messages=messages,
            temperature=0.7,
            stop=user_stop_words
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"API调用失败：{str(e)}"


# 新增ElevenLabs工具类
class ElevenLabsTTSTool:
    name = "elevenlabs_tts"
    description = "将文本转换为语音文件（使用ElevenLabs API）"
    args = {
        "text": {
            "type": "string",
            "description": "需要转换为语音的文本内容"
        },
        "filename": {
            "type": "string",
            "description": "保存语音文件的路径（默认：static/audio/speech.mp3）",
            "default": "static/audio/speech.mp3"
        }
    }

    def invoke(self, input_args):
        try:
            if isinstance(input_args, str):
                input_args = json.loads(input_args)

            text = input_args.get("text")
            filename = input_args.get("filename", "static/audio/speech.mp3")

            # 生成唯一文件名
            timestamp = int(datetime.datetime.now().timestamp())
            filename = f"static/audio/speech_{timestamp}.mp3"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            from elevenlabs.client import ElevenLabs
            from elevenlabs import save
            client = ElevenLabs()
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            save(audio, filename)

            # 返回带播放按钮的Markdown
            url_path = f"/{filename}"
            return f"语音生成成功：\n[点击播放音频]({url_path})"
        except Exception as e:
            return f"语音生成失败：{str(e)}"


# 新增ArXiv检索工具类
class ArxivSearchTool:
    name = "arxiv_search"
    description = "检索学术论文（使用ArXiv API）"
    args = {
        "query": {
            "type": "string",
            "description": "搜索关键词，支持布尔逻辑和字段过滤（如：ti:deep learning AND au:yoshua）"
        },
        "max_results": {
            "type": "number",
            "description": "返回的最大结果数（默认：5）",
            "default": 5
        }
    }

    def invoke(self, input_args):
        try:
            if isinstance(input_args, str):
                input_args = json.loads(input_args)

            query = input_args.get("query")
            max_results = input_args.get("max_results", 5)

            # 执行ArXiv搜索
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            results = []
            for result in client.results(search):
                results.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category
                })

            if not results:
                return "未找到相关论文"

            # 格式化输出（限制长度避免token超限）
            output = []
            for i, paper in enumerate(results[:3], 1):  # 最多显示3篇
                # 修改ArxivSearchTool的invoke方法
                output.append(
                    f"{i}. 【{paper['title']}】\n"
                    f"作者：{', '.join(paper['authors'][:3])}{'等' if len(paper['authors']) > 3 else ''}\n"
                    f"摘要：{paper['summary'][:150]}...\n"
                    f"PDF链接：{paper['pdf_url']}\n"  # 明确标识PDF链接
                    f"发布时间：{paper['published']}"
                )
            return "\n\n".join(output)

        except Exception as e:
            return f"论文检索失败：{str(e)}"


# 新增图片生成工具类
class ImageGenerationTool:
    name = "image_generation"
    description = "根据文字描述生成图片（使用DashScope API）"
    args = {
        "prompt": {
            "type": "string",
            "description": "图片生成的文字描述（必须使用中文描述）"
        },
        "n": {
            "type": "number",
            "description": "生成图片数量（默认：1）",
            "default": 1
        },
        "size": {
            "type": "string",
            "description": "图片尺寸（默认：1024*1024，可选：512*512, 1024*1024, 720*1280, 1280*720）",
            "default": "1024*1024"
        },
        "output_dir": {
            "type": "string",
            "description": "图片保存目录（默认：static/images/）",
            "default": "static/images/"
        },
        "base_filename": {
            "type": "string",
            "description": "图片基础名称（默认：generated_image）",
            "default": "generated_image"
        }
    }

    # 修改ImageGenerationTool的invoke方法
    def invoke(self, input_args):
        try:
            if isinstance(input_args, str):
                input_args = json.loads(input_args)

            prompt = input_args.get("prompt")
            n = input_args.get("n", 1)
            size = input_args.get("size", "1024*1024")
            output_dir = input_args.get("output_dir", "static/images/")
            base_filename = input_args.get("base_filename", "generated_image")

            rsp = ImageSynthesis.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model="wanx2.1-t2i-turbo",
                prompt=prompt,
                n=n,
                size=size
            )

            if rsp.status_code == HTTPStatus.OK:
                os.makedirs(output_dir, exist_ok=True)
                saved_files = []

                for index, result in enumerate(rsp.output.results):
                    original_file = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                    _, ext = os.path.splitext(original_file)
                    filename = f"{base_filename}_{index}{ext}"
                    save_path = os.path.join(output_dir, filename)
                    with open(save_path, 'wb+') as f:
                        f.write(requests.get(result.url).content)
                    url_path = f"/static/images/{filename}"
                    saved_files.append(url_path)

                # 返回Markdown格式的图片链接
                image_markdown = "\n".join([f"![]({path})" for path in saved_files])
                return f"图片生成成功：\n{image_markdown}"
            else:
                return f"图片生成失败：{rsp.code} - {rsp.message}"

        except Exception as e:
            return f"图片生成异常：{str(e)}"


# 初始化工具列表（新增Arxiv工具）
os.environ['TAVILY_API_KEY'] = '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'
tavily = TavilySearchResults(max_results=5)
tavily.description = '这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'

arxiv_tool = ArxivSearchTool()
elevenlabs_tts = ElevenLabsTTSTool()
image_gen_tool = ImageGenerationTool()
tools = [tavily, elevenlabs_tts, arxiv_tool, image_gen_tool]  # 添加新工具到列表
tool_names = 'or'.join([tool.name for tool in tools])

# 工具描述生成（保持不变）
tool_descs = []
for t in tools:
    #工具参数描述
    args_desc = []
    #获取工具中各参数名称与信息
    for name, info in t.args.items():
        #添加工具所需参数的名称、描述和类型
        args_desc.append({
            'name': name,
            'description': info.get('description', ''),
            'type': info['type']
        })
    #参数描述存入json
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append(f'{t.name}: {t.description}, args: {args_desc}')
tool_descs = '\n'.join(tool_descs)

# 提示模板（优化版）
prompt_tpl = '''当前日期：{today}
历史对话：
{chat_history}

可用工具：
{tool_descs}

请使用以下格式响应：

Question: 需要回答的问题
Thought: 思考过程
Action: 选择工具（从[{tool_names}]中选择）
Action Input: 工具输入
Observation: 工具返回结果
...（可重复多次）

当得到最终答案时：
Thought: 最终结论
Final Answer: 最终答案

现在开始！

Question: {query}
{agent_scratchpad}'''


def agent_execute(query, chat_history=[]):
    agent_scratchpad = ''
    while True:
        # 构建提示
        history = '\n'.join([f'用户：{his[0]}\n助手：{his[1]}' for his in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(
            today=today,
            chat_history=history,
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=query,
            agent_scratchpad=agent_scratchpad
        )


        # 调用大模型
        response = llm(prompt, user_stop_words=['Observation:'])
        # 解析响应
        final_answer_i = response.rfind('Final Answer:')
        action_i = response.rfind('Action:')

        # 处理最终答案
        if final_answer_i != -1:
            final_answer = response[final_answer_i + len('Final Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history

        # 解析工具调用
        action_input_i = response.rfind('Action Input:')
        if action_i == -1 or action_input_i == -1:
            return False, '响应格式错误', chat_history

        action = response[action_i + len('Action:'):action_input_i].strip()
        action_input = response[action_input_i + len('Action Input:'):].strip()

        # 执行工具调用
        the_tool = next((t for t in tools if t.name == action), None)
        if not the_tool:
            observation = '工具不存在'
            agent_scratchpad += f"{response}\nObservation: {observation}\n"
            continue

        try:
            action_input = json.loads(action_input)
            observation = the_tool.invoke(action_input)
            print("模型工具输出：" + str(observation))
        except Exception as e:
            observation = f'工具执行失败：{str(e)}'

        agent_scratchpad += f"{response}\nObservation: {str(observation)}\n"


# 重试机制（保持不变）
def agent_execute_with_retry(query, chat_history=[], retry_times=3):
    for _ in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history


# 交互界面
if __name__ == "__main__":
    my_history = []
    print("输入 'exit' 结束对话")
    while True:
        query = input('\n用户：')
        if query.lower() == 'exit':
            break
        _, response, my_history = agent_execute_with_retry(query, my_history)
        print(f"\n助手：{response}")
        my_history = my_history[-5:]  # 保留最近5轮对话