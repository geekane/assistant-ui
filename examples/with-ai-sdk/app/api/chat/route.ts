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
    const { messages, system, tools: clientToolsPayload } = await req.json(); // 从请求中获取原始的 tools

    const modelId = "deepseek-ai/DeepSeek-R1";

    // 准备 streamText 函数的选项
    const streamTextOptions: {
      model: any; // 建议使用从 'ai' 或 '@ai-sdk/provider' 导入的 LanguageModelV1 类型
      messages: import('ai').CoreMessage[]; // 明确 messages 的类型
      system?: string;
      tools?: Record<string, import('ai').CoreTool>;
      toolChoice?: 'auto' | 'none' | 'required' | { type: 'tool'; toolName: string }; // <<< 已更正：name -> toolName
      // toolCallStreaming?: boolean;
      // 其他 streamText 支持的参数...
    } = {
      model: modelscopeProvider(modelId),
      messages, // 确保 messages 是 CoreMessage[] 类型
      // system 和 tools 会在下面条件性地添加到这个对象上
    };

    // 只有当有有效的工具时，才添加 tools 属性到选项中
    if (clientToolsPayload) {
      const actualTools: Record<string, import('ai').CoreTool> = {
        ...frontendTools(clientToolsPayload), // 确保 frontendTools 返回的结构是 Record<string, CoreTool>
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

      // 确保 actualTools 不是空对象才赋值
      if (Object.keys(actualTools).length > 0) {
        streamTextOptions.tools = actualTools;
        // 如果需要，在这里一同设置 toolChoice 或 toolCallStreaming
        // streamTextOptions.toolCallStreaming = true;
      }
    }

    const result = await streamText(streamTextOptions);

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
