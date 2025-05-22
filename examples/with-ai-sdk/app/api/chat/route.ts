import { createOpenAI } from "@ai-sdk/openai"; // 导入 createOpenAI
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import { streamText } from "ai"; // 移除了未使用的 experimental_streamText
import { z } from "zod";

// 声明 Edge Runtime 和最大持续时间
export const runtime = "edge";
export const maxDuration = 30; // 单位是秒

// --- ModelScope 配置 ---
// 1. 从环境变量定义您的 API 密钥和 Base URL
//    如果环境变量未设置，则使用明确的字符串后备值。
//    警告：强烈建议在 Cloudflare Pages 的环境变量中设置您自己的有效密钥。
//    这里的后备 API 密钥仅作为类型占位符，不应是生产中使用的真实密钥。
const MODELSCOPE_API_KEY = process.env['MODELSCOPE_API_KEY'] || "YOUR_MODELSCOPE_API_KEY_PLACEHOLDER";
const MODELSCOPE_BASE_URL = process.env['MODELSCOPE_BASE_URL'] || "https://api-inference.modelscope.cn/v1/";

// 2. 为 ModelScope 创建一个自定义的 OpenAI provider 实例
//    确保这里的 apiKey 和 baseURL 始终是 string 类型，以满足 createOpenAI 的参数要求。
const modelscopeProvider = createOpenAI({
  apiKey: MODELSCOPE_API_KEY,
  baseURL: MODELSCOPE_BASE_URL,
  // 如果 ModelScope 需要自定义请求头，您可以在此添加 headers 选项。
  // 例如: headers: { 'Authorization': `Bearer ${MODELSCOPE_API_KEY}` }
  // 但通常 createOpenAI 会自动处理标准的 Bearer token。
  // 如果 ModelScope 的认证方式非常特殊，可能需要更复杂的配置。
});
// --- ModelScope 配置结束 ---

export async function POST(req: Request) {
  try {
    const { messages, system, tools } = await req.json();

    // 3. 指定 ModelScope 的模型 ID
    const modelId = "deepseek-ai/DeepSeek-R1"; // 或者您想使用的其他 ModelScope 模型 ID

    // 4. 使用 streamText 进行调用，传入自定义的 provider 和模型 ID
    const result = await streamText({
      model: modelscopeProvider(modelId),
      messages,
      // 根据 ModelScope 是否支持以及如何支持工具调用流来决定是否启用或调整 tools 部分
      // toolCallStreaming: true, // 如果 ModelScope 不支持或格式不同，可能需要注释或修改
      system,
      tools: tools ? { // 确保 tools 不为 undefined 才展开
        ...frontendTools(tools), // 这个函数转换的工具格式需要与 ModelScope 兼容
        // 示例工具，请根据实际需求调整或移除
        weather: {
          description: "Get weather information",
          parameters: z.object({
            location: z.string().describe("Location to get weather for"),
          }),
          execute: async ({ location }: { location:string }) => {
            // 这是一个模拟的工具执行，实际应调用天气 API
            return `The weather in ${location} is sunny.`;
          },
        },
      } : undefined,
      // 如果需要，可以添加其他参数，如 temperature, max_tokens 等
      // temperature: 0.7,
      // maxTokens: 1024,
    });

    return result.toDataStreamResponse();

  } catch (error) {
    console.error("[API CHAT ERROR]", error);
    // 返回一个标准的错误响应
    if (error instanceof Error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ error: "An unknown error occurred" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
