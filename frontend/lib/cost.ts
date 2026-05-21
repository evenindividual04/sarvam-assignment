// Mirror of utils/cost_model.py. All zeros for free-tier providers.

export const COST_PER_1K_TOKENS: Record<
  string,
  { prompt: number; completion: number }
> = {
  "gemini-2.5-flash": { prompt: 0.0, completion: 0.0 },
  "groq-llama-3.3-70b": { prompt: 0.0, completion: 0.0 },
  "gpt-4o-mini-github": { prompt: 0.0, completion: 0.0 },
  "openrouter-deepseek-r1": { prompt: 0.0, completion: 0.0 },
};

export const DEFAULT_MODEL = "gemini-2.5-flash";

export function costFor(
  model: string | undefined,
  promptTokens: number | undefined,
  completionTokens: number | undefined,
): number {
  const rates = COST_PER_1K_TOKENS[model ?? DEFAULT_MODEL];
  if (!rates) return 0;
  const p = (promptTokens ?? 0) / 1000;
  const c = (completionTokens ?? 0) / 1000;
  return p * rates.prompt + c * rates.completion;
}

export function formatCost(usd: number): string {
  return `$${usd.toFixed(4)}`;
}
