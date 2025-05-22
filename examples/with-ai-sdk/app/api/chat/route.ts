import { createOpenAI } from "@ai-sdk/openai";
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import {
  streamText,
  type CoreTool,
  type CoreMessage,
  type LanguageModelV1, // Vercel AI SDK 的模型类型
  type ToolChoice, // Vercel AI SDK 的 ToolChoice 类型
} from "ai";
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
const modelscopeProvider: LanguageModelV1 = createOpenAI({ // 明确 modelscopeProvider 的类型
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
    // 为请求体添加更明确的类型
    const { messages, system, tools: clientToolsPayload } = await req.json() as {
      messages: CoreMessage[];
      system?: string;
      tools?: any; // 根据实际前端发送的 tools 结构来定义更准确的类型，或保持 any 如果不确定
    };

    const modelId = "deepseek-ai/DeepSeek-R1"; // 或者您想使用的其他 ModelScope 模型 ID

    // 准备工具定义
    let actualToolDefinitions: Record<string, CoreTool> | undefined = undefined;

    if (clientToolsPayload) {
      const combinedTools: Record<string, CoreTool> = {
        // 确保 frontendTools 返回的结构是 Record<string, CoreTool>
        // 如果 clientToolsPayload 本身就是期望的工具格式，则可能不需要 frontendTools
        ...(typeof frontendTools === 'function' ? frontendTools(clientToolsPayload) : clientToolsPayload),
        // 您定义的 weather 工具
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

      // 确保 combinedTools 不是空对象才赋值
      if (Object.keys(combinedTools).length > 0) {
        actualToolDefinitions = combinedTools;
      }
    }

    // 使用条件展开来构造传递给 streamText 的参数对象
    // 注意：我们不再预先定义 streamTextOptions 的完整类型，让 TypeScript 根据实际传递的属性推断
    const result = await streamText({
      model: modelscopeProvider(modelId), // modelscopeProvider(modelId) 通常返回一个可直接使用的模型实例
      messages,
      ...(system && { system }), // 只有当 system 存在时才包含它
      ...(actualToolDefinitions && { tools: actualToolDefinitions }), // 只有当 actualToolDefinitions 存在时才包含 tools
      // 如果您需要明确指定 toolChoice，可以这样条件性添加：
      // ...(actualToolDefinitions && { toolChoice: 'auto' as ToolChoice<Record<string, CoreTool>> }), // 例如 'auto'
      // 或者如果要指定特定工具：
      // ...(actualToolDefinitions && { toolChoice: { type: 'tool', toolName: 'weather' } as ToolChoice<Record<string, CoreTool>> }),
      // toolCallStreaming 也可以类似地条件性添加
      // ...(actualToolDefinitions && { toolCallStreaming: true }),
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
