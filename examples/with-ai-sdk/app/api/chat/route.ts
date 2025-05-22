import { createOpenAI } from "@ai-sdk/openai";
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import {
  streamText,
  type CoreTool,
  type CoreMessage,
  type LanguageModelV1, // 这个类型导入本身是好的，因为它用于 streamText 的 model 参数
} from "ai";
import { z } from "zod";

// 声明 Edge Runtime 和最大持续时间
export const runtime = "edge";
export const maxDuration = 30; // 单位是秒

// --- ModelScope 配置 ---
const MODELSCOPE_API_KEY = process.env['MODELSCOPE_API_KEY'] || "YOUR_MODELSCOPE_API_KEY_PLACEHOLDER";
const MODELSCOPE_BASE_URL = process.env['MODELSCOPE_BASE_URL'] || "https://api-inference.modelscope.cn/v1/";

// 2. 为 ModelScope 创建一个自定义的 OpenAI provider 实例
//    让 TypeScript 自动推断 modelscopeProvider 的类型 (它将是 OpenAIProvider 类型)
const modelscopeProvider = createOpenAI({
  apiKey: MODELSCOPE_API_KEY,
  baseURL: MODELSCOPE_BASE_URL,
});
// --- ModelScope 配置结束 ---

export async function POST(req: Request) {
  try {
    const { messages, system, tools: clientToolsPayload } = await req.json() as {
      messages: CoreMessage[];
      system?: string;
      tools?: any;
    };

    const modelId = "deepseek-ai/DeepSeek-R1";

    // 准备工具定义
    let actualToolDefinitions: Record<string, CoreTool> | undefined = undefined;

    if (clientToolsPayload) {
      const combinedTools: Record<string, CoreTool> = {
        ...(typeof frontendTools === 'function' ? frontendTools(clientToolsPayload) : clientToolsPayload),
        weather: {
          description: "Get weather information",
          parameters: z.object({
            location: z.string().describe("Location to get weather for"),
          }),
          execute: async ({ location }: { location: string }) => {
            return `The weather in ${location} is sunny.`;
          },
        },
      };

      if (Object.keys(combinedTools).length > 0) {
        actualToolDefinitions = combinedTools;
      }
    }

    // 调用 streamText 时，modelscopeProvider(modelId) 会返回 LanguageModelV1 类型的实例
    const result = await streamText({
      model: modelscopeProvider(modelId), // 正确：这里传入的是模型实例
      messages,
      ...(system && { system }),
      ...(actualToolDefinitions && { tools: actualToolDefinitions }),
    });

    return result.toDataStreamResponse();

  } catch (error) {
    console.error("[API CHAT ERROR]", error);
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
