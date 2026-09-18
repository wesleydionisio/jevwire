/**
 * The OpenRouter provider, end to end without a network.
 *
 * Resolution rules, model naming, the wire format, error mapping, and — the
 * part that matters most — that a hook judging through OpenRouter fails open
 * exactly as it does through TypeSafe, and that no key ever reaches a report or
 * the decision log.
 */

import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig, missingKeyMessage } from "../src/config.js";
import { createJevModel, JevDecisionModel } from "../src/jev/client.js";
import {
  JevAuthError,
  JevConnectionError,
  JevOverloadedError,
  JevProtocolError,
  JevRateLimitError,
  JevTimeoutError,
} from "../src/jev/errors.js";
import {
  OPENROUTER_BASE_URL,
  OPENROUTER_LATEST_MODEL,
  resolveModel,
  resolveProvider,
} from "../src/jev/provider.js";
import { authorize, credentials, expectedKeysFrom } from "../src/hooks/daemon/auth.js";
import { loadHookConfig } from "../src/hooks/config.js";
import { hookConfigFrom, sessionConfigOf } from "../src/hooks/daemon/registry.js";
import { handlePreToolUse } from "../src/hooks/handlers/pre-tool-use.js";
import { handleSessionStart } from "../src/hooks/handlers/session-start.js";
import { statusReport } from "../src/hooks/report.js";
import { modelCost, Store } from "../src/hooks/store.js";
import type { HookInput } from "../src/hooks/types.js";
import type { Question } from "../src/decision/types.js";
import { cleanup, makeDeps, tempDir, testConfig } from "./hooks/helpers.js";

const OR_KEY = "sk-or-v1-secret-openrouter-key-0123456789";
const TS_KEY = "sk-typesafe-secret-key-0123456789";

// ---------------------------------------------------------------- fake fetch

interface Recorded {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

function fakeFetch(respond: () => Response | Promise<Response>): { fetch: typeof fetch; requests: Recorded[] } {
  const requests: Recorded[] = [];
  const impl = async (input: Parameters<typeof fetch>[0], init?: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries((init?.headers ?? {}) as Record<string, string>)) {
      headers[key.toLowerCase()] = value;
    }
    requests.push({
      url: String(input),
      method: init?.method ?? "GET",
      headers,
      body: typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : undefined,
    });
    return await respond();
  };
  return { fetch: impl as unknown as typeof fetch, requests };
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

const OK_BODY = {
  model: "typesafe/jev-1.13",
  answers: {
    urgent: { type: "noul", noul: 0.92 },
    team: {
      type: "choice",
      choice: "billing",
      probabilities: { billing: 0.8, technical: 0.15, other: 0.05 },
      confidence: 0.71,
    },
    mood: { type: "score", score: 1.4, legend: { "0": "calm", "1": "annoyed", "2": "angry" }, probabilities: { "0": 0.1, "1": 0.5, "2": 0.4 }, confidence: 0.6 },
  },
  usage: { input_tokens: 321, output_tokens: 12 },
};

/** Identity helper: keeps each question's literal type while checking it against the contract. */
const questions = <T extends Record<string, Question>>(value: T): T => value;

const QUESTIONS = questions({
  urgent: { type: "noul", instructions: "Is it urgent?" },
  team: { type: "choice", instructions: "Which team?", criteria: { billing: null, technical: null, other: null } },
  mood: { type: "score", instructions: "How upset?", criteria: ["calm", "annoyed", "angry"] },
});

function openrouter(fetchImpl: typeof fetch, overrides: Partial<ConstructorParameters<typeof JevDecisionModel>[0]> = {}) {
  return new JevDecisionModel({
    apiKey: OR_KEY,
    provider: "openrouter",
    fetch: fetchImpl,
    sleep: async () => {},
    maxRetries: 0,
    ...overrides,
  });
}

// ------------------------------------------------------- provider resolution

describe("resolveProvider", () => {
  it("auto-detects TypeSafe from TYPESAFE_API_KEY", () => {
    const r = resolveProvider({ typesafeKey: TS_KEY });
    expect(r).toMatchObject({ provider: "typesafe", apiKey: TS_KEY, setting: "auto" });
  });

  it("auto-detects OpenRouter when only OPENROUTER_API_KEY exists", () => {
    const r = resolveProvider({ openrouterKey: OR_KEY });
    expect(r).toMatchObject({ provider: "openrouter", apiKey: OR_KEY, setting: "auto" });
  });

  it("prefers TypeSafe under auto when both keys exist, so existing installs never change", () => {
    expect(resolveProvider({ typesafeKey: TS_KEY, openrouterKey: OR_KEY }).provider).toBe("typesafe");
    expect(resolveProvider({ setting: "auto", typesafeKey: TS_KEY, openrouterKey: OR_KEY }).provider).toBe("typesafe");
  });

  it("honours an explicit provider over key precedence", () => {
    expect(resolveProvider({ setting: "openrouter", typesafeKey: TS_KEY, openrouterKey: OR_KEY })).toMatchObject({
      provider: "openrouter",
      apiKey: OR_KEY,
    });
    expect(resolveProvider({ setting: "TypeSafe", typesafeKey: TS_KEY, openrouterKey: OR_KEY })).toMatchObject({
      provider: "typesafe",
      apiKey: TS_KEY,
    });
  });

  it("never falls back silently when the explicit provider has no key", () => {
    const or = resolveProvider({ setting: "openrouter", typesafeKey: TS_KEY });
    expect(or.provider).toBeNull();
    expect(or.apiKey).toBeNull();
    expect(or.problem).toContain("OPENROUTER_API_KEY");

    const ts = resolveProvider({ setting: "typesafe", openrouterKey: OR_KEY });
    expect(ts.provider).toBeNull();
    expect(ts.problem).toContain("TYPESAFE_API_KEY");
  });

  it("is inactive with no keys at all, and ignores empty or unsubstituted values", () => {
    expect(resolveProvider({}).provider).toBeNull();
    expect(resolveProvider({ typesafeKey: "  ", openrouterKey: "${user_config.openrouter_api_key}" }).provider).toBeNull();
  });

  it("treats an unknown setting as auto and says so", () => {
    const r = resolveProvider({ setting: "azure", openrouterKey: OR_KEY });
    expect(r.provider).toBe("openrouter");
    expect(r.problem).toContain("azure");
  });
});

describe("resolveModel", () => {
  it("leaves TypeSafe names untouched", () => {
    expect(resolveModel("typesafe", undefined)).toBe("jev-1.13.0");
    expect(resolveModel("typesafe", "jev-latest")).toBe("jev-latest");
    expect(resolveModel("typesafe", "jev-1.12.0")).toBe("jev-1.12.0");
  });

  it("maps jev-latest to the OpenRouter release", () => {
    expect(resolveModel("openrouter", "jev-latest")).toBe(OPENROUTER_LATEST_MODEL);
    expect(OPENROUTER_LATEST_MODEL).toBe("typesafe/jev-1.13");
  });

  it("defaults to typesafe/jev-1.13, including from the plugin's own jev-1.13.0 default", () => {
    expect(resolveModel("openrouter", undefined)).toBe("typesafe/jev-1.13");
    expect(resolveModel("openrouter", "jev-1.13.0")).toBe("typesafe/jev-1.13");
  });

  it("prefixes a bare version and passes a full or custom slug through", () => {
    expect(resolveModel("openrouter", "jev-1.12")).toBe("typesafe/jev-1.12");
    expect(resolveModel("openrouter", "typesafe/jev-1.13")).toBe("typesafe/jev-1.13");
    expect(resolveModel("openrouter", "someone/custom-jev-2.0")).toBe("someone/custom-jev-2.0");
  });
});

// ----------------------------------------------------------------- MCP config

describe("loadConfig with OpenRouter", () => {
  it("selects OpenRouter from its key alone", () => {
    const config = loadConfig({ OPENROUTER_API_KEY: OR_KEY });
    expect(config).toMatchObject({
      provider: "openrouter",
      apiKey: OR_KEY,
      baseUrl: OPENROUTER_BASE_URL,
      model: "typesafe/jev-1.13",
    });
  });

  it("honours JEV_PROVIDER and JEV_MODEL", () => {
    const config = loadConfig({
      JEV_PROVIDER: "openrouter",
      OPENROUTER_API_KEY: OR_KEY,
      TYPESAFE_API_KEY: TS_KEY,
      JEV_MODEL: "typesafe/jev-1.13",
    });
    expect(config.provider).toBe("openrouter");
    expect(config.apiKey).toBe(OR_KEY);
    expect(config.model).toBe("typesafe/jev-1.13");
  });

  it("does not let TYPESAFE_BASE_URL leak into OpenRouter", () => {
    const config = loadConfig({ OPENROUTER_API_KEY: OR_KEY, TYPESAFE_BASE_URL: "http://localhost:9000" });
    expect(config.baseUrl).toBe(OPENROUTER_BASE_URL);
  });

  it("keeps a TypeSafe-only environment exactly as it was", () => {
    const config = loadConfig({ TYPESAFE_API_KEY: TS_KEY, TYPESAFE_BASE_URL: "http://localhost:9000/" });
    expect(config).toMatchObject({ provider: "typesafe", baseUrl: "http://localhost:9000", model: "jev-1.13.0" });
  });

  it("lets the plugin's provider option win, but not when it is 'auto'", () => {
    const both = { OPENROUTER_API_KEY: OR_KEY, TYPESAFE_API_KEY: TS_KEY };
    expect(loadConfig({ ...both, CLAUDE_PLUGIN_OPTION_PROVIDER: "openrouter" }).provider).toBe("openrouter");
    expect(loadConfig({ ...both, CLAUDE_PLUGIN_OPTION_PROVIDER: "auto", JEV_PROVIDER: "openrouter" }).provider).toBe("openrouter");
    expect(loadConfig({ ...both, CLAUDE_PLUGIN_OPTION_PROVIDER: "${user_config.provider}" }).provider).toBe("typesafe");
  });

  it("reads the plugin's OpenRouter key option", () => {
    const config = loadConfig({ JEV_PLUGIN_OPENROUTER_API_KEY: OR_KEY, CLAUDE_PLUGIN_OPTION_PROVIDER: "openrouter" });
    expect(config.provider).toBe("openrouter");
    expect(config.apiKey).toBe(OR_KEY);
  });

  it("represents an explicit provider without its key, and explains it", () => {
    const config = loadConfig({ JEV_PROVIDER: "openrouter", TYPESAFE_API_KEY: TS_KEY });
    expect(config.provider).toBeNull();
    expect(config.apiKey).toBeNull();
    expect(createJevModel(config)).toBeNull();
    const message = missingKeyMessage(config);
    expect(message).toContain("OPENROUTER_API_KEY");
    expect(message).toContain("does not fall back");
    expect(message).not.toContain(TS_KEY);
  });

  it("keeps the historical message when nothing names a provider", () => {
    expect(missingKeyMessage(loadConfig({}))).toContain("TYPESAFE_API_KEY");
  });
});

// ---------------------------------------------------------------------- client

describe("OpenRouter transport", () => {
  it("posts {model,state,questions} to the Decisions API with a Bearer key", async () => {
    const { fetch, requests } = fakeFetch(() => json(OK_BODY));
    const model = openrouter(fetch, { model: "jev-latest" });

    await model.evaluate({ state: "Help, payouts failed", questions: QUESTIONS });

    expect(requests).toHaveLength(1);
    const req = requests[0]!;
    expect(req.url).toBe("https://openrouter.ai/api/alpha/decisions");
    expect(req.method).toBe("POST");
    expect(req.headers.authorization).toBe(`Bearer ${OR_KEY}`);
    expect(req.headers["content-type"]).toBe("application/json");
    expect(req.headers["x-title"]).toBe("jevwire");
    expect(req.headers["http-referer"]).toContain("github.com");
    expect(req.body).toEqual({ model: "typesafe/jev-1.13", state: "Help, payouts failed", questions: QUESTIONS });
  });

  it("sends a per-request model through resolveModel and a custom slug as given", async () => {
    const { fetch, requests } = fakeFetch(() => json(OK_BODY));
    const model = openrouter(fetch);
    await model.evaluate({ state: "s", questions: QUESTIONS, model: "jev-latest" });
    await model.evaluate({ state: "s", questions: QUESTIONS, model: "typesafe/jev-1.12" });
    expect(requests.map((r) => (r.body as { model: string }).model)).toEqual(["typesafe/jev-1.13", "typesafe/jev-1.12"]);
  });

  it("does not send OpenRouter headers to TypeSafe, and keeps the System One URL", async () => {
    const { fetch, requests } = fakeFetch(() => json({ ...OK_BODY, model: "jev-1.13.0" }));
    const model = new JevDecisionModel({ apiKey: TS_KEY, fetch, maxRetries: 0 });
    await model.evaluate({ state: "s", questions: QUESTIONS });
    expect(requests[0]!.url).toBe("https://api.typesafe.ai/v1/systemone");
    expect(requests[0]!.headers["x-title"]).toBeUndefined();
    expect(requests[0]!.headers.authorization).toBe(`Bearer ${TS_KEY}`);
    expect((requests[0]!.body as { model: string }).model).toBe("jev-latest");
  });

  it("adapts the response: answers, probabilities, confidence, usage, model and provider", async () => {
    const { fetch } = fakeFetch(() => json(OK_BODY));
    const result = await openrouter(fetch).evaluate({ state: "s", questions: QUESTIONS });

    expect(result.provider).toBe("openrouter");
    expect(result.model).toBe("typesafe/jev-1.13");
    expect(result.usage).toEqual({ input_tokens: 321, output_tokens: 12 });
    expect(result.answers.urgent.noul).toBe(0.92);
    expect(result.answers.team.choice).toBe("billing");
    expect(result.answers.team.probabilities).toEqual({ billing: 0.8, technical: 0.15, other: 0.05 });
    expect(result.answers.team.confidence).toBe(0.71);
    expect(result.answers.mood.score).toBe(1.4);
    expect(result.answers.mood.confidence).toBe(0.6);
  });

  it("tolerates a response with no usage block, as the alpha endpoint may omit it", async () => {
    const { usage: _usage, ...noUsage } = OK_BODY;
    const { fetch } = fakeFetch(() => json(noUsage));
    const result = await openrouter(fetch).evaluate({ state: "s", questions: QUESTIONS });
    expect(result.usage).toEqual({ input_tokens: 0, output_tokens: 0 });
  });

  it("falls back to the requested slug when the response names no model", async () => {
    const { model: _model, ...noModel } = OK_BODY;
    const { fetch } = fakeFetch(() => json(noModel));
    const result = await openrouter(fetch, { model: "typesafe/jev-1.13" }).evaluate({ state: "s", questions: QUESTIONS });
    expect(result.model).toBe("typesafe/jev-1.13");
  });

  it("lists its one model without a request", async () => {
    const { fetch, requests } = fakeFetch(() => json({}));
    const catalog = await openrouter(fetch).listModels();
    expect(catalog.models.map((m) => m.name)).toEqual(["typesafe/jev-1.13"]);
    expect(requests).toHaveLength(0);
  });
});

describe("OpenRouter error handling", () => {
  const one = (response: () => Response | Promise<Response>) =>
    openrouter(fakeFetch(response).fetch).evaluate({ state: "s", questions: QUESTIONS });

  it("maps 401 to an auth error that names the right key and never echoes it", async () => {
    const error = await one(() => json({ error: { message: "bad key" } }, 401)).catch((e: unknown) => e as Error);
    expect(error).toBeInstanceOf(JevAuthError);
    expect((error as Error).message).toContain("OpenRouter");
    expect((error as Error).message).toContain("OPENROUTER_API_KEY");
    expect(JSON.stringify(error)).not.toContain(OR_KEY);
  });

  it("maps 429 to a rate-limit error", async () => {
    await expect(one(() => json({}, 429))).rejects.toBeInstanceOf(JevRateLimitError);
  });

  it("maps 5xx to an overloaded error", async () => {
    await expect(one(() => json({}, 503))).rejects.toBeInstanceOf(JevOverloadedError);
    await expect(one(() => json({}, 500))).rejects.toBeInstanceOf(JevOverloadedError);
  });

  it("maps a network failure to a connection error", async () => {
    await expect(
      one(() => {
        throw new TypeError("fetch failed");
      }),
    ).rejects.toBeInstanceOf(JevConnectionError);
  });

  it("maps a slow response to a timeout", async () => {
    const slow = ((_input: unknown, init?: RequestInit) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
      })) as unknown as typeof fetch;
    await expect(
      openrouter(slow, { timeoutMs: 20 }).evaluate({ state: "s", questions: QUESTIONS }),
    ).rejects.toBeInstanceOf(JevTimeoutError);
  });

  it("rejects malformed bodies as protocol errors", async () => {
    await expect(one(() => new Response("<html>", { status: 200 }))).rejects.toBeInstanceOf(JevProtocolError);
    await expect(one(() => json({ answers: {} }))).rejects.toBeInstanceOf(JevProtocolError);
    await expect(
      one(() => json({ answers: { urgent: { type: "choice" }, team: OK_BODY.answers.team, mood: OK_BODY.answers.mood } })),
    ).rejects.toBeInstanceOf(JevProtocolError);
  });

  it("retries a 429 in MCP mode but makes exactly one request with the hooks' zero retries", async () => {
    const mcp = fakeFetch(() => json({}, 429));
    await expect(
      openrouter(mcp.fetch, { maxRetries: 2 }).evaluate({ state: "s", questions: QUESTIONS }),
    ).rejects.toBeInstanceOf(JevRateLimitError);
    expect(mcp.requests).toHaveLength(3);

    const hook = fakeFetch(() => json({}, 429));
    const config = loadHookConfig({ OPENROUTER_API_KEY: OR_KEY });
    expect(config.maxRetries).toBe(0);
    await expect(
      openrouter(hook.fetch, { maxRetries: config.maxRetries }).evaluate({ state: "s", questions: QUESTIONS }),
    ).rejects.toBeInstanceOf(JevRateLimitError);
    expect(hook.requests).toHaveLength(1);
  });
});

// ----------------------------------------------------------------------- hooks

describe("hook configuration", () => {
  it("auto-detects OpenRouter and pins the hooks' short timeout and zero retries", () => {
    const config = loadHookConfig({ OPENROUTER_API_KEY: OR_KEY });
    expect(config).toMatchObject({
      provider: "openrouter",
      apiKey: OR_KEY,
      baseUrl: OPENROUTER_BASE_URL,
      model: "typesafe/jev-1.13",
      timeoutMs: 1500,
      maxRetries: 0,
    });
  });

  it("reads the plugin's options first, and an explicit TypeSafe choice ignores the OpenRouter key", () => {
    const config = loadHookConfig({
      CLAUDE_PLUGIN_OPTION_PROVIDER: "typesafe",
      CLAUDE_PLUGIN_OPTION_API_KEY: TS_KEY,
      OPENROUTER_API_KEY: OR_KEY,
    });
    expect(config).toMatchObject({ provider: "typesafe", apiKey: TS_KEY, model: "jev-1.13.0" });
  });

  it("is inactive, not switched, when the explicit provider has no key", () => {
    const config = loadHookConfig({ JEV_PROVIDER: "openrouter", TYPESAFE_API_KEY: TS_KEY });
    expect(config.provider).toBeNull();
    expect(config.apiKey).toBeNull();
    expect(createJevModel(config)).toBeNull();
  });

  it("keeps the provider in the daemon's hands, not the session snapshot", () => {
    const config = loadHookConfig({ OPENROUTER_API_KEY: OR_KEY });
    const snapshot = sessionConfigOf(config);
    expect(snapshot).not.toHaveProperty("provider");
    expect(snapshot).not.toHaveProperty("apiKey");
    expect(hookConfigFrom(snapshot, OR_KEY, "/tmp/d", config).provider).toBe("openrouter");
  });
});

describe("daemon credentials", () => {
  it("accepts the OpenRouter key from either new header", () => {
    const keys = expectedKeysFrom({ OPENROUTER_API_KEY: OR_KEY });
    expect(keys).toEqual([OR_KEY]);
    expect(authorize({ "x-jev-env-key-openrouter": OR_KEY }, keys)).toBe(true);
    expect(authorize({ "x-jev-option-key-openrouter": OR_KEY }, keys)).toBe(true);
    expect(authorize({ "x-jev-env-key-openrouter": "wrong" }, keys)).toBe(false);
  });

  it("treats empty OpenRouter headers as absent, like the TypeSafe ones", () => {
    expect(credentials({ authorization: "Bearer ", "x-jev-env-key-openrouter": "" })).toEqual([]);
  });

  it("holds keys from both providers so a rotated half still works", () => {
    expect(expectedKeysFrom({ TYPESAFE_API_KEY: TS_KEY, OPENROUTER_API_KEY: OR_KEY })).toEqual([TS_KEY, OR_KEY]);
  });
});

describe("manifests", () => {
  const root = new URL("../", import.meta.url).pathname;
  const plugin = JSON.parse(readFileSync(`${root}plugin/.claude-plugin/plugin.json`, "utf8")) as {
    userConfig: Record<string, { options?: string[]; sensitive?: boolean; default?: unknown }>;
    mcpServers: { jev: { env: Record<string, string> } };
  };

  it("offers provider and OpenRouter key options, sensitive and with no default that could shadow JEV_PROVIDER", () => {
    expect(plugin.userConfig.provider?.options).toEqual(["auto", "typesafe", "openrouter"]);
    expect(plugin.userConfig.provider?.default).toBeUndefined();
    expect(plugin.userConfig.openrouter_api_key?.sensitive).toBe(true);
    expect(plugin.userConfig.api_key?.sensitive).toBe(true);
  });

  it("exports them to the MCP server", () => {
    const env = plugin.mcpServers.jev.env;
    expect(env.CLAUDE_PLUGIN_OPTION_PROVIDER).toBe("${user_config.provider}");
    expect(env.CLAUDE_PLUGIN_OPTION_OPENROUTER_API_KEY).toBe("${user_config.openrouter_api_key}");
    expect(env.JEV_PLUGIN_OPENROUTER_API_KEY).toBe("${user_config.openrouter_api_key}");
  });
});

describe("hooks through OpenRouter", () => {
  let dir: string;
  beforeEach(() => {
    dir = tempDir();
  });
  afterEach(() => {
    cleanup(dir);
  });

  const input: HookInput = {
    session_id: "s1",
    cwd: "/home/dev/project",
    permission_mode: "default",
    hook_event_name: "PreToolUse",
    tool_name: "Bash",
    tool_input: { command: "curl -X POST https://example.com/pay" },
    tool_use_id: "toolu_1",
  };

  function rig(fetchImpl: typeof fetch) {
    const config = testConfig(dir, loadHookConfig({ CLAUDE_PLUGIN_DATA: dir, OPENROUTER_API_KEY: OR_KEY }));
    const model = new JevDecisionModel({
      apiKey: OR_KEY,
      provider: "openrouter",
      model: config.model,
      timeoutMs: config.timeoutMs,
      maxRetries: config.maxRetries,
      fetch: fetchImpl,
    });
    const deps = makeDeps(dir, { model, config });
    deps.store.updateSession("s1", (s) => ({ ...s, prompts: ["fix the failing date parser test"] }));
    return deps;
  }

  const failures: [string, () => Response | Promise<Response>][] = [
    ["429", () => json({}, 429, { "retry-after": "1" })],
    ["503", () => json({}, 503)],
    ["401", () => json({ error: { message: "nope" } }, 401)],
    ["malformed body", () => new Response("not json", { status: 200 })],
    ["empty answers", () => json({ answers: {} })],
    [
      "network error",
      () => {
        throw new TypeError("fetch failed");
      },
    ],
  ];

  for (const [name, respond] of failures) {
    it(`fails open on ${name}: no output, no allow, one request, error logged without the key`, async () => {
      const { fetch, requests } = fakeFetch(respond);
      const deps = rig(fetch);

      const output = await handlePreToolUse(input, deps);

      expect(output).toBeUndefined();
      expect(requests).toHaveLength(1);
      const record = deps.store.readLog().at(-1);
      expect(record?.decision).toBe("error");
      expect(JSON.stringify(deps.store.readLog())).not.toContain(OR_KEY);
    });
  }

  it("fails open on a timeout well inside the hook's wall clock", async () => {
    const hang = ((_input: unknown, init?: RequestInit) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
      })) as unknown as typeof fetch;
    const deps = rig(hang);
    deps.config.timeoutMs = 50;
    const model = new JevDecisionModel({
      apiKey: OR_KEY,
      provider: "openrouter",
      timeoutMs: 50,
      maxRetries: 0,
      fetch: hang,
    });
    const started = Date.now();
    expect(await handlePreToolUse(input, { ...deps, model })).toBeUndefined();
    expect(Date.now() - started).toBeLessThan(1500);
  });

  it("records the answering provider in the decision log, and omits it when unknown", () => {
    expect(
      modelCost({ model: "typesafe/jev-1.13", provider: "openrouter", latency_ms: 5, usage: { input_tokens: 1 } }),
    ).toMatchObject({ model: "typesafe/jev-1.13", provider: "openrouter" });
    expect(modelCost({ model: "jev-1.13.0", latency_ms: 5, usage: { input_tokens: 1 } })).not.toHaveProperty("provider");
  });

  it("SessionStart names OpenRouter when it was chosen and its key is missing, and never suggests a silent switch", async () => {
    const config = loadHookConfig({ CLAUDE_PLUGIN_DATA: dir, JEV_PROVIDER: "openrouter", TYPESAFE_API_KEY: TS_KEY });
    const deps = makeDeps(dir, { config });
    const output = await handleSessionStart({ session_id: "s9", hook_event_name: "SessionStart" } as HookInput, deps);
    const text = JSON.stringify(output);
    expect(text).toContain("OpenRouter");
    expect(text).toContain("OPENROUTER_API_KEY");
    expect(text).not.toContain(TS_KEY);
  });
});

describe("/jev:status", () => {
  let dir: string;
  beforeEach(() => {
    dir = tempDir();
  });
  afterEach(() => {
    cleanup(dir);
  });

  it("shows provider and model for OpenRouter and never the key", () => {
    const config = loadHookConfig({ CLAUDE_PLUGIN_DATA: dir, OPENROUTER_API_KEY: OR_KEY });
    const report = statusReport(config, new Store(dir), 1_700_000_000_000);
    expect(report).toContain("provider: OpenRouter");
    expect(report).toContain("model: typesafe/jev-1.13");
    expect(report).toContain("API key: configured");
    expect(report).toContain("gate: advisory");
    expect(report).not.toContain(OR_KEY);
    expect(report).not.toContain("sk-or-");
  });

  it("shows TypeSafe for an unchanged install", () => {
    const config = loadHookConfig({ CLAUDE_PLUGIN_DATA: dir, TYPESAFE_API_KEY: TS_KEY });
    const report = statusReport(config, new Store(dir), 1_700_000_000_000);
    expect(report).toContain("provider: TypeSafe");
    expect(report).toContain("model: jev-1.13.0");
    expect(report).not.toContain(TS_KEY);
  });

  it("says why there is no provider, without naming a key", () => {
    const config = loadHookConfig({ CLAUDE_PLUGIN_DATA: dir, JEV_PROVIDER: "openrouter", TYPESAFE_API_KEY: TS_KEY });
    const report = statusReport(config, new Store(dir), 1_700_000_000_000);
    expect(report).toContain("provider: none");
    expect(report).toContain("OPENROUTER_API_KEY is not set");
    expect(report).toContain("not configured");
    expect(report).not.toContain(TS_KEY);
  });
});
