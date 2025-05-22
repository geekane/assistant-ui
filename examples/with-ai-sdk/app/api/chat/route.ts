import { createOpenAI } from "@ai-sdk/openai"; // 导入 createOpenAI
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import { streamText } from "ai";
import { z } from "zod";

export const runtime = "edge";
export const maxDuration = 30;

// --- ModelScope 配置 ---
// 1. 定义您的 API 密钥和 Base URL
// 强烈建议使用环境变量存储敏感信息
const MODELSCOPE_API_KEY = process.env['MODELSCOPE_API_KEY'];
const MODELSCOPE_BASE_URL = process.env['MODELSCOPE_BASE_URL'];

// 2. 为 ModelScope 创建一个自定义的 OpenAI provider 实例
const modelscopeProvider = createOpenAI({
  apiKey: MODELSCOPE_API_KEY,
  baseURL: MODELSCOPE_BASE_URL,
  // 如果 ModelScope 需要自定义请求头，您可能需要在这里添加。
  // 您的 Python 示例中没有显示需要额外的请求头。
  // headers: { 'Custom-Header': 'value' },

  // Vercel AI SDK 的 createOpenAI 通常会自动添加 "Authorization: Bearer <token>" 请求头。
  // 如果 ModelScope 期望 API 密钥以不同的方式传递（例如在不同的请求头中），
  // 则可能需要更高级的自定义，例如通过 `compatibility: 'manual'` 并自己处理 fetch 请求。
  // 但从您的 Python 示例来看，它使用了标准的 OpenAI Python 客户端，
  // 这通常意味着标准的 Bearer token 认证方式是兼容的。
});

// --- ModelScope 配置结束 ---

export async function POST(req: Request) {
  const { messages, system, tools } = await req.json();

  // 3. 指定 ModelScope 的模型 ID
  const modelId = "deepseek-ai/DeepSeek-R1"; // 或者您想使用的其他 ModelScope 模型 ID

  // 使用 streamText 进行调用
  const result = await streamText({
    // 4. 使用自定义的 provider 和指定的模型
    model: modelscopeProvider(modelId), // 将模型 ID 传递给 provider
    messages,
    // toolCallStreaming: true, // 根据 ModelScope 是否支持以及如何支持工具调用流来决定是否启用
    system,
    tools: tools ? { // 确保 tools 不为 undefined 才展开
      ...frontendTools(tools), // 这个函数可能需要检查是否与 ModelScope 的工具格式兼容
      weather: {
        description: "Get weather information",
        parameters: z.object({
          location: z.string().describe("Location to get weather for"),
        }),
        execute: async ({ location }: { location: string }) => { // 为 execute 的参数添加类型
          return `The weather in ${location} is sunny.`;
        },
      },
    } : undefined, // 如果 tools 为空，则传递 undefined
    // 您可能需要根据 ModelScope API 的具体要求调整其他参数，
    // 例如 temperature, max_tokens 等。这些可以作为 streamText 的额外参数传入。
    // maxTokens: 1024,
    // temperature: 0.7,
  });

  return result.toDataStreamResponse();
}
