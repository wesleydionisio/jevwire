/**
 * Which service answers a Jev judgment, and under what model name.
 *
 * The decision contract (`src/decision/types.ts`) does not care: TypeSafe's
 * `POST /v1/systemone` and OpenRouter's Decisions API
 * (`POST /api/alpha/decisions`) take the same `{model, state, questions}` body
 * and return the same `answers`. What differs is the URL, the model slug, and a
 * few headers — so a provider is a small value here, not a second client.
 *
 * Everything in this file is pure and dependency-free: it is imported by the
 * hook bundle, which must stay small, and by tests that never touch a network.
 */

export const PROVIDERS = ["typesafe", "openrouter"] as const;
export type ProviderName = (typeof PROVIDERS)[number];

/** What the user asked for. `auto` means "decide from the keys present". */
export type ProviderSetting = ProviderName | "auto";

export const TYPESAFE_BASE_URL = "https://api.typesafe.ai";
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/alpha";

/** Pinned TypeSafe default; see `DEFAULTS.model` in `src/config.ts` for why. */
export const TYPESAFE_DEFAULT_MODEL = "jev-1.13.0";

/**
 * OpenRouter has no redirecting "latest" slug, only pinned versions under the
 * `typesafe/` vendor prefix. `jev-latest` therefore means "the current release"
 * here, and moving it is a one-line change.
 */
export const OPENROUTER_LATEST_MODEL = "typesafe/jev-1.13";
export const OPENROUTER_DEFAULT_MODEL = OPENROUTER_LATEST_MODEL;

/** Sent to OpenRouter so usage is attributed to this project in their dashboards. */
export const OPENROUTER_APP_TITLE = "jevwire";
export const OPENROUTER_APP_REFERER = "https://github.com/Brainwires/jevwire";

export const PROVIDER_LABELS: Record<ProviderName, string> = {
  typesafe: "TypeSafe",
  openrouter: "OpenRouter",
};

/** The environment variable that holds each provider's key, for messages. */
export const PROVIDER_KEY_VARS: Record<ProviderName, string> = {
  typesafe: "TYPESAFE_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
};

export interface ProviderInputs {
  /** Raw `JEV_PROVIDER` / plugin option value, if any. Case-insensitive. */
  setting?: string | undefined;
  /** Already-resolved key for each provider, or undefined. Callers own their lookup order. */
  typesafeKey?: string | undefined;
  openrouterKey?: string | undefined;
}

export interface ProviderResolution {
  /** What the user asked for, normalized. An unrecognized value becomes `auto`. */
  setting: ProviderSetting;
  /**
   * The provider that will be used, or `null` when there is nothing to call:
   * no key anywhere under `auto`, or an explicit provider whose key is missing.
   * A `null` provider is inactive — never a silent switch to the other one.
   */
  provider: ProviderName | null;
  /** The selected provider's key. `null` exactly when `provider` is `null`. */
  apiKey: string | null;
  /**
   * The provider a base URL, model default and message should be written for
   * when `provider` is null: the explicit one if named, else TypeSafe (the
   * historical default).
   */
  target: ProviderName;
  /** Why nothing is selected, or a note about a setting that was ignored. */
  problem: string | null;
}

function clean(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed === undefined || trimmed === "" || trimmed.includes("${") ? undefined : trimmed;
}

/**
 * Pick the provider.
 *
 * 1. An explicit setting is honoured, and never falls back: `openrouter` with
 *    no OpenRouter key is "no provider", not "use TypeSafe instead".
 * 2. Otherwise `TYPESAFE_API_KEY` wins, so an existing install is unchanged.
 * 3. Otherwise `OPENROUTER_API_KEY`.
 * 4. Otherwise nothing: hooks stay inactive, the server explains what to set.
 */
export function resolveProvider(inputs: ProviderInputs): ProviderResolution {
  const typesafeKey = clean(inputs.typesafeKey);
  const openrouterKey = clean(inputs.openrouterKey);
  const rawSetting = clean(inputs.setting)?.toLowerCase();

  let setting: ProviderSetting = "auto";
  let problem: string | null = null;
  if (rawSetting === "typesafe" || rawSetting === "openrouter") {
    setting = rawSetting;
  } else if (rawSetting !== undefined && rawSetting !== "auto") {
    problem = `JEV_PROVIDER=${JSON.stringify(rawSetting)} is not one of auto|typesafe|openrouter; using auto.`;
  }

  const keys: Record<ProviderName, string | undefined> = { typesafe: typesafeKey, openrouter: openrouterKey };

  if (setting !== "auto") {
    const key = keys[setting];
    if (key === undefined) {
      return {
        setting,
        provider: null,
        apiKey: null,
        target: setting,
        problem: `JEV_PROVIDER=${setting} but ${PROVIDER_KEY_VARS[setting]} is not set.`,
      };
    }
    return { setting, provider: setting, apiKey: key, target: setting, problem };
  }

  if (typesafeKey !== undefined) return { setting, provider: "typesafe", apiKey: typesafeKey, target: "typesafe", problem };
  if (openrouterKey !== undefined) {
    return { setting, provider: "openrouter", apiKey: openrouterKey, target: "openrouter", problem };
  }
  return { setting, provider: null, apiKey: null, target: "typesafe", problem };
}

/** The base URL a provider talks to when nothing overrides it. */
export function defaultBaseUrl(provider: ProviderName): string {
  return provider === "openrouter" ? OPENROUTER_BASE_URL : TYPESAFE_BASE_URL;
}

/** The model a provider uses when the user named none. */
export function defaultModel(provider: ProviderName): string {
  return provider === "openrouter" ? OPENROUTER_DEFAULT_MODEL : TYPESAFE_DEFAULT_MODEL;
}

/**
 * The single place a user-facing model name becomes a wire model name.
 *
 * TypeSafe: passed through untouched, so nothing changes for existing installs.
 *
 * OpenRouter: `jev-latest` → the current release; a bare `jev-X.Y[.Z]` gains the
 * `typesafe/` vendor prefix; `typesafe/...` (or any other `vendor/...`) is used
 * as given. A three-part version drops its patch number, because OpenRouter
 * names releases `jev-1.13` and the plugin's own default is `jev-1.13.0` — an
 * OpenRouter user who never touched the model setting must still get a real slug.
 */
export function resolveModel(provider: ProviderName, requested: string | undefined): string {
  const name = clean(requested) ?? defaultModel(provider);
  if (provider !== "openrouter") return name;

  if (name.toLowerCase() === "jev-latest" || name.toLowerCase() === "typesafe/jev-latest") {
    return OPENROUTER_LATEST_MODEL;
  }
  if (name.includes("/")) return name;
  return `typesafe/${name.replace(/^(jev-\d+\.\d+)\.\d+$/i, "$1")}`;
}

/**
 * Result fragment naming the answering provider; empty when the model does not
 * report one. Lives here, not in `tools/shared.ts`, because that file imports
 * zod and the hook bundle must not.
 */
export function providerField(provider: string | undefined): { provider?: string } {
  return provider === undefined ? {} : { provider };
}
