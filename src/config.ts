/**
 * Environment parsing.
 *
 * The server must boot without an API key: an MCP client typically starts the
 * server before the user has finished configuring it, and a process that exits
 * at startup gives them a transport error instead of an explanation. So a
 * missing key is recorded here (`apiKey: null`) and surfaced as an `isError`
 * tool result that says what to set.
 */

import type { GateThresholds } from "./decision/types.js";
import { resolveProjectRoot } from "./files/index.js";
import {
  defaultBaseUrl,
  PROVIDER_KEY_VARS,
  PROVIDER_LABELS,
  resolveModel,
  resolveProvider,
  type ProviderName,
  type ProviderSetting,
} from "./jev/provider.js";

export interface Config {
  /**
   * The selected provider's key. `null` when there is none (no key under
   * `auto`, or an explicit provider whose key is unset) — the server still starts.
   */
  apiKey: string | null;
  /** The provider that will be called, or `null` when none is configured. */
  provider: ProviderName | null;
  /** What `JEV_PROVIDER` asked for (`auto` when unset). */
  providerSetting: ProviderSetting;
  /** Why no provider is active, or a note about an ignored setting. `null` otherwise. */
  providerProblem: string | null;
  baseUrl: string;
  /** The model name as sent on the wire (OpenRouter: `typesafe/jev-1.13`). */
  model: string;
  timeoutMs: number;
  maxRetries: number;
  thresholds: GateThresholds;
  maxConcurrency: number;
  /**
   * True when `JEV_MAX_CONCURRENCY` was set. File-source ranking fans out
   * wider than the default, but an operator who named a number meant it.
   */
  maxConcurrencyExplicit: boolean;
  /** Project root the file-reading tools are confined to. */
  projectRoot: string;
}

export const DEFAULTS = {
  baseUrl: "https://api.typesafe.ai",
  /**
   * Pinned: an alias move would shift every probability under thresholds the
   * caller has tuned. `JEV_MODEL=jev-latest` opts back into following
   * releases. The library's own default (`src/jev/client.ts`) stays the alias —
   * a library user picks their own policy.
   */
  model: "jev-1.13.0",
  timeoutMs: 30_000,
  maxRetries: 3,
  autoThreshold: 0.85,
  reviewThreshold: 0.6,
  maxConcurrency: 4,
} as const;

/**
 * Message used wherever a call needs a key that was never configured.
 * `missingKeyMessage(config)` narrows it when a provider was named explicitly.
 */
export const MISSING_API_KEY_MESSAGE =
  "TYPESAFE_API_KEY is not set, so this server cannot reach the Jev API. " +
  "Set it in the MCP server's environment (for example: " +
  "`claude mcp add jev -e TYPESAFE_API_KEY=sk-... -- npx -y jevwire`) and restart the server. " +
  "If this is the Claude Code plugin, either set the API key in the plugin's settings or export " +
  "TYPESAFE_API_KEY before starting Claude Code, then run /reload-plugins. " +
  "Keys are issued at https://typesafe.ai. " +
  "To use OpenRouter instead, set OPENROUTER_API_KEY (and optionally JEV_PROVIDER=openrouter).";

/** The message for a call that has no provider to run on. */
export function missingKeyMessage(config: Pick<Config, "providerProblem" | "providerSetting">): string {
  if (config.providerSetting === "auto") return MISSING_API_KEY_MESSAGE;
  const provider = config.providerSetting;
  return (
    `${config.providerProblem ?? `${PROVIDER_KEY_VARS[provider]} is not set.`} ` +
    `${PROVIDER_LABELS[provider]} was selected explicitly, so jev does not fall back to another provider. ` +
    `Set ${PROVIDER_KEY_VARS[provider]} (or the plugin's matching API key option), or change JEV_PROVIDER, then restart.`
  );
}

export type Env = Record<string, string | undefined>;

function readString(env: Env, key: string, fallback: string): string {
  const raw = env[key];
  if (raw === undefined) return fallback;
  const trimmed = raw.trim();
  return trimmed === "" ? fallback : trimmed;
}

function readNumber(env: Env, key: string, fallback: number, min: number, max: number): number {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(
      `${key}=${JSON.stringify(raw)} is not usable: expected a number between ${min} and ${max}.`,
    );
  }
  return value;
}

/**
 * The plugin manifest passes its `api_key` option as JEV_PLUGIN_API_KEY rather
 * than as TYPESAFE_API_KEY: when the option is left empty, a manifest entry
 * named TYPESAFE_API_KEY would overwrite a key the user exported in their shell
 * with an empty string. An empty or unsubstituted (`${…}`) value is skipped.
 */
function firstKey(...candidates: (string | undefined)[]): string | undefined {
  for (const candidate of candidates) {
    const value = candidate?.trim();
    if (value !== undefined && value !== "" && !value.includes("${")) return value;
  }
  return undefined;
}

/**
 * Build a `Config` from an environment. Throws only on a *malformed* value —
 * a missing API key is represented, not thrown.
 */
export function loadConfig(env: Env = process.env): Config {
  const typesafeKey = firstKey(env.JEV_PLUGIN_API_KEY, env.CLAUDE_PLUGIN_OPTION_API_KEY, env.TYPESAFE_API_KEY);
  const openrouterKey = firstKey(
    env.JEV_PLUGIN_OPENROUTER_API_KEY,
    env.CLAUDE_PLUGIN_OPTION_OPENROUTER_API_KEY,
    env.OPENROUTER_API_KEY,
  );
  // The plugin option's "auto" must not shadow a JEV_PROVIDER the user exported.
  const settingRaw = [env.CLAUDE_PLUGIN_OPTION_PROVIDER, env.JEV_PROVIDER]
    .map((value) => firstKey(value))
    .find((value) => value !== undefined && value.toLowerCase() !== "auto");
  const resolved = resolveProvider({ setting: settingRaw ?? firstKey(env.JEV_PROVIDER), typesafeKey, openrouterKey });
  const auto = readNumber(env, "JEV_AUTO_THRESHOLD", DEFAULTS.autoThreshold, 0, 1);
  const review = readNumber(env, "JEV_REVIEW_THRESHOLD", DEFAULTS.reviewThreshold, 0, 1);

  if (review > auto) {
    throw new Error(
      `JEV_REVIEW_THRESHOLD (${review}) must not exceed JEV_AUTO_THRESHOLD (${auto}).`,
    );
  }

  const baseUrl =
    resolved.target === "openrouter"
      ? readString(env, "OPENROUTER_BASE_URL", defaultBaseUrl("openrouter"))
      : readString(env, "TYPESAFE_BASE_URL", DEFAULTS.baseUrl);

  return {
    apiKey: resolved.apiKey,
    provider: resolved.provider,
    providerSetting: resolved.setting,
    providerProblem: resolved.problem,
    baseUrl: baseUrl.replace(/\/+$/, ""),
    // The plugin manifest exports its `model` option here too, so the setting
    // means the same thing to the server as it does to the hooks. An empty or
    // unsubstituted option falls through to the env var and then the default.
    // `resolveModel` is the one place a name becomes a wire name: identity for
    // TypeSafe, the `typesafe/...` slug for OpenRouter.
    model: resolveModel(
      resolved.target,
      firstKey(env.CLAUDE_PLUGIN_OPTION_MODEL) ?? readString(env, "JEV_MODEL", DEFAULTS.model),
    ),
    timeoutMs: readNumber(env, "JEV_TIMEOUT_MS", DEFAULTS.timeoutMs, 1, 600_000),
    maxRetries: readNumber(env, "JEV_MAX_RETRIES", DEFAULTS.maxRetries, 0, 10),
    thresholds: { auto, review },
    maxConcurrency: readNumber(env, "JEV_MAX_CONCURRENCY", DEFAULTS.maxConcurrency, 1, 32),
    maxConcurrencyExplicit: (env.JEV_MAX_CONCURRENCY ?? "").trim() !== "",
    projectRoot: resolveProjectRoot(env),
  };
}
