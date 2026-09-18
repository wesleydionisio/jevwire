/**
 * HTTP client for Jev as a `DecisionModel`: TypeSafe's System One endpoint, or
 * OpenRouter's Decisions API (same body, same answers; see `provider.ts`).
 *
 * There is no TypeSafe SDK dependency here: the wire shapes in
 * `src/decision/types.ts` mirror `POST /v1/systemone` closely enough that this
 * is a passthrough plus four things the raw API does not do for you —
 * retry/backoff, deadlines, local budget and schema checks, and verification
 * that every question you asked came back answered with the right type.
 */

import {
  checkBudget,
  DEFAULT_BUDGET_LIMITS,
  type BudgetLimits,
} from "../decision/budget.js";
import type { ListModelsResult, ModelCatalog, ModelInfo } from "../decision/models.js";
import { validateQuestions } from "../decision/validate.js";
import type {
  Answer,
  ChoiceAnswer,
  ChoiceQuestion,
  DecisionModel,
  EvaluateRequest,
  EvaluateResult,
  NoulQuestion,
  Question,
  ScoreAnswer,
  ScoreQuestion,
  State,
  Usage,
} from "../decision/types.js";
import {
  JevAuthError,
  JevConnectionError,
  JevError,
  JevOverloadedError,
  JevProtocolError,
  JevRateLimitError,
  JevTimeoutError,
  JevValidationError,
} from "./errors.js";
import {
  defaultBaseUrl,
  OPENROUTER_APP_REFERER,
  OPENROUTER_APP_TITLE,
  OPENROUTER_LATEST_MODEL,
  PROVIDER_KEY_VARS,
  PROVIDER_LABELS,
  resolveModel,
  type ProviderName,
} from "./provider.js";

export type { ListModelsResult, ModelInfo } from "../decision/models.js";

export interface JevDecisionModelOptions {
  apiKey: string;
  /** Which service to call. Defaults to `typesafe`, so existing callers are unchanged. */
  provider?: ProviderName;
  /** Origin (TypeSafe) or API root (OpenRouter). Defaults to the provider's own. */
  baseUrl?: string;
  /** Default model for requests that do not override it. */
  model?: string;
  /** Deadline for one logical call, covering all retries. */
  timeoutMs?: number;
  /** Retries *after* the first attempt. 0 disables retrying. */
  maxRetries?: number;
  /** Injected for tests. Defaults to global `fetch`. */
  fetch?: typeof fetch;
  /** Injected for tests so backoff waits do not take real time. */
  sleep?: (ms: number) => Promise<void>;
  /** Override the context limits, e.g. for a future model with a bigger window. */
  budgetLimits?: BudgetLimits;
}

const DEFAULT_MODEL = "jev-latest";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;
const BACKOFF_BASE_MS = 500;
const BACKOFF_CAP_MS = 10_000;

/** Full-jitter exponential backoff, capped. Exported for tests. */
export function computeBackoffMs(attempt: number, random: () => number = Math.random): number {
  const ceiling = Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** attempt);
  return Math.round(ceiling * (0.5 + 0.5 * random()));
}

/**
 * Parse a `retry-after` header: delay-seconds or an HTTP date.
 * Returns milliseconds, clamped to the backoff cap, or `null` if unusable.
 */
export function parseRetryAfter(header: string | null, now: number = Date.now()): number | null {
  if (header === null) return null;
  const trimmed = header.trim();
  if (trimmed === "") return null;

  if (/^\d+(\.\d+)?$/.test(trimmed)) {
    const ms = Number(trimmed) * 1000;
    return Math.min(Math.max(ms, 0), BACKOFF_CAP_MS);
  }

  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) return null;
  return Math.min(Math.max(date - now, 0), BACKOFF_CAP_MS);
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

export class JevDecisionModel implements DecisionModel, ModelCatalog {
  /** The model name as sent on the wire (for OpenRouter, the `typesafe/...` slug). */
  readonly name: string;
  readonly provider: ProviderName;

  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly budgetLimits: BudgetLimits;

  constructor(options: JevDecisionModelOptions) {
    this.apiKey = options.apiKey;
    this.provider = options.provider ?? "typesafe";
    this.baseUrl = (options.baseUrl ?? defaultBaseUrl(this.provider)).replace(/\/+$/, "");
    this.name = resolveModel(this.provider, options.model ?? (this.provider === "typesafe" ? DEFAULT_MODEL : undefined));
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.sleep = options.sleep ?? defaultSleep;
    this.budgetLimits = options.budgetLimits ?? DEFAULT_BUDGET_LIMITS;

    if (typeof this.fetchImpl !== "function") {
      throw new JevError("No fetch implementation available. Node 20+ or an injected `fetch` is required.");
    }
  }

  async evaluate<Q extends Record<string, Question>>(request: EvaluateRequest<Q>): Promise<EvaluateResult<Q>> {
    const questions = request.questions as Record<string, Question>;
    validateQuestions(questions);
    checkBudget(request.state, questions, this.budgetLimits);

    const model = request.model === undefined ? this.name : resolveModel(this.provider, request.model);
    const started = Date.now();

    const body = await this.send(
      this.provider === "openrouter" ? "/decisions" : "/v1/systemone",
      { state: request.state, model, questions },
      request.signal,
    );

    const latency_ms = Date.now() - started;
    const parsed = this.parseEvaluateResponse(body, questions);

    return {
      model: parsed.model,
      provider: this.provider,
      answers: parsed.answers as EvaluateResult<Q>["answers"],
      usage: parsed.usage,
      latency_ms,
    };
  }

  async choice(state: State, question: Omit<ChoiceQuestion, "type">): Promise<ChoiceAnswer> {
    const result = await this.evaluate({ state, questions: { q: { type: "choice", ...question } } });
    return result.answers.q;
  }

  async score(state: State, question: Omit<ScoreQuestion, "type">): Promise<ScoreAnswer> {
    const result = await this.evaluate({ state, questions: { q: { type: "score", ...question } } });
    return result.answers.q;
  }

  async probability(state: State, question: Omit<NoulQuestion, "type">): Promise<number> {
    const result = await this.evaluate({ state, questions: { q: { type: "noul", ...question } } });
    return result.answers.q.noul;
  }

  /**
   * `GET /v1/models` — the names this account may send in `model`.
   *
   * OpenRouter's Decisions API has no catalog endpoint, so there the answer is
   * the one model this package targets, stated as what it is.
   */
  async listModels(signal?: AbortSignal): Promise<ListModelsResult> {
    if (this.provider === "openrouter") {
      return {
        models: [
          {
            name: OPENROUTER_LATEST_MODEL,
            description:
              "Jev on OpenRouter's Decisions API. OpenRouter publishes no model listing for it; " +
              "`jev-latest` maps to this slug.",
            release_date: "",
          },
        ],
      };
    }
    const body = await this.send("/v1/models", undefined, signal);
    if (typeof body !== "object" || body === null || !Array.isArray((body as { models?: unknown }).models)) {
      throw new JevProtocolError("GET /v1/models did not return a `models` array.", { body });
    }
    return { models: (body as { models: ModelInfo[] }).models };
  }

  // ---------------------------------------------------------------- internals

  /**
   * One logical request: attempts, backoff and the deadline all live here.
   * Returns the parsed JSON body of a 2xx response.
   */
  private async send(path: string, payload: unknown, callerSignal?: AbortSignal): Promise<unknown> {
    const deadline = Date.now() + this.timeoutMs;
    let lastError: JevError | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      this.throwIfCallerAborted(callerSignal);

      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        throw lastError instanceof JevTimeoutError
          ? lastError
          : new JevTimeoutError(`Request to ${path} exceeded the ${this.timeoutMs}ms timeout.`, {
              cause: lastError,
            });
      }

      let response: Response;
      try {
        response = await this.attempt(path, payload, callerSignal, remaining);
      } catch (error) {
        if (error instanceof JevError && !(error instanceof JevConnectionError)) throw error;
        if (!(error instanceof JevError)) throw error;
        lastError = error;
        if (attempt >= this.maxRetries) throw error;
        await this.backoff(attempt, null, deadline, callerSignal);
        continue;
      }

      if (response.ok) {
        return await this.readJson(response, path);
      }

      const error = await this.toError(response, path);

      // 401 and 422 are caller mistakes; retrying just burns the deadline.
      if (!isRetryableStatus(response.status)) throw error;

      lastError = error;
      if (attempt >= this.maxRetries) throw error;
      await this.backoff(attempt, response.headers.get("retry-after"), deadline, callerSignal);
    }

    /* c8 ignore next */
    throw lastError ?? new JevError(`Request to ${path} failed with no attempts made.`);
  }

  /** A single HTTP attempt, with its own abort plumbing. */
  private async attempt(
    path: string,
    payload: unknown,
    callerSignal: AbortSignal | undefined,
    remainingMs: number,
  ): Promise<Response> {
    const controller = new AbortController();
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, remainingMs);

    const onCallerAbort = (): void => {
      controller.abort();
    };
    callerSignal?.addEventListener("abort", onCallerAbort, { once: true });

    try {
      const init: RequestInit = {
        method: payload === undefined ? "GET" : "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          Accept: "application/json",
          ...(payload === undefined ? {} : { "Content-Type": "application/json" }),
          ...(this.provider === "openrouter"
            ? {
                "HTTP-Referer": OPENROUTER_APP_REFERER,
                "X-Title": OPENROUTER_APP_TITLE,
                "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
              }
            : {}),
        },
        signal: controller.signal,
      };
      if (payload !== undefined) init.body = JSON.stringify(payload);

      return await this.fetchImpl(`${this.baseUrl}${path}`, init);
    } catch (error) {
      if (isAbortError(error) || controller.signal.aborted) {
        this.throwIfCallerAborted(callerSignal);
        if (timedOut) {
          throw new JevTimeoutError(`Request to ${path} exceeded the ${this.timeoutMs}ms timeout.`, {
            cause: error,
          });
        }
      }
      throw new JevConnectionError(
        `Could not reach ${this.baseUrl}${path}: ${error instanceof Error ? error.message : String(error)}`,
        { cause: error },
      );
    } finally {
      clearTimeout(timer);
      callerSignal?.removeEventListener("abort", onCallerAbort);
    }
  }

  private throwIfCallerAborted(signal: AbortSignal | undefined): void {
    if (signal?.aborted === true) {
      throw signal.reason ?? new JevTimeoutError("Request aborted by the caller.");
    }
  }

  private async backoff(
    attempt: number,
    retryAfter: string | null,
    deadline: number,
    callerSignal: AbortSignal | undefined,
  ): Promise<void> {
    const hinted = parseRetryAfter(retryAfter);
    const wait = Math.min(hinted ?? computeBackoffMs(attempt), Math.max(deadline - Date.now(), 0));
    if (wait > 0) await this.sleep(wait);
    this.throwIfCallerAborted(callerSignal);
  }

  private async readJson(response: Response, path: string): Promise<unknown> {
    const text = await response.text();
    try {
      return JSON.parse(text) as unknown;
    } catch (error) {
      throw new JevProtocolError(`${path} returned a non-JSON body.`, {
        status: response.status,
        body: text.slice(0, 500),
        cause: error,
      });
    }
  }

  private async toError(response: Response, path: string): Promise<JevError> {
    let body: unknown;
    try {
      const text = await response.text();
      try {
        body = JSON.parse(text) as unknown;
      } catch {
        body = text.slice(0, 500);
      }
    } catch {
      body = undefined;
    }

    const options = { status: response.status, body };
    const label = PROVIDER_LABELS[this.provider];

    switch (response.status) {
      case 401:
        return new JevAuthError(
          `${label} rejected the API key. Check ${PROVIDER_KEY_VARS[this.provider]}.`,
          options,
        );
      case 422:
        return new JevValidationError(
          `${label} rejected the request body as invalid; the response names the offending field.`,
          options,
        );
      case 429:
        return new JevRateLimitError(`${label} rate limit exceeded.`, options);
      case 529:
        return new JevOverloadedError(`${label} is temporarily overloaded.`, options);
      default:
        if (response.status >= 500) {
          return new JevOverloadedError(`${label} returned a server error on ${path}.`, options);
        }
        return new JevError(`${label} returned an unexpected status on ${path}.`, options);
    }
  }

  /**
   * Check the response against the questions we asked. An id we never asked
   * about is ignored; a missing id, or one whose answer `type` disagrees with
   * the question, is a protocol error — silently handing a caller the wrong
   * answer shape is worse than failing.
   */
  private parseEvaluateResponse(
    body: unknown,
    questions: Record<string, Question>,
  ): { model: string; answers: Record<string, Answer>; usage: Usage } {
    if (typeof body !== "object" || body === null) {
      throw new JevProtocolError("Evaluate response was not a JSON object.", { body });
    }

    const raw = body as { model?: unknown; answers?: unknown; usage?: unknown };

    if (typeof raw.answers !== "object" || raw.answers === null || Array.isArray(raw.answers)) {
      throw new JevProtocolError("Evaluate response is missing the `answers` object.", { body });
    }
    const answers = raw.answers as Record<string, Answer | undefined>;

    const missing: string[] = [];
    const mismatched: string[] = [];

    for (const [id, question] of Object.entries(questions)) {
      const answer = answers[id];
      if (answer === undefined || answer === null) {
        missing.push(id);
        continue;
      }
      if (answer.type !== question.type) {
        mismatched.push(`${id} (asked ${question.type}, got ${JSON.stringify(answer.type)})`);
      }
    }

    if (missing.length > 0) {
      throw new JevProtocolError(
        `Evaluate response is missing answers for: ${missing.join(", ")}.`,
        { body },
      );
    }
    if (mismatched.length > 0) {
      throw new JevProtocolError(
        `Evaluate response answer types do not match the questions asked: ${mismatched.join("; ")}.`,
        { body },
      );
    }

    const usage = raw.usage as Partial<Usage> | undefined;

    return {
      model: typeof raw.model === "string" ? raw.model : this.name,
      answers: answers as Record<string, Answer>,
      usage: {
        input_tokens: typeof usage?.input_tokens === "number" ? usage.input_tokens : 0,
        output_tokens: typeof usage?.output_tokens === "number" ? usage.output_tokens : 0,
      },
    };
  }
}

/** The slice of `Config` / `HookConfig` that decides which client to build. */
export interface ModelSource {
  apiKey: string | null;
  provider: ProviderName | null;
  baseUrl: string;
  model: string;
  timeoutMs: number;
  maxRetries: number;
}

/**
 * The one place a resolved configuration becomes a client, so the MCP server,
 * the hook CLI, the daemon and the smoke test cannot drift apart. `null` means
 * "no usable provider" — the caller stays inactive rather than guessing one.
 */
export function createJevModel(source: ModelSource): JevDecisionModel | null {
  if (source.apiKey === null || source.provider === null) return null;
  return new JevDecisionModel({
    apiKey: source.apiKey,
    provider: source.provider,
    baseUrl: source.baseUrl,
    model: source.model,
    timeoutMs: source.timeoutMs,
    maxRetries: source.maxRetries,
  });
}

function isRetryableStatus(status: number): boolean {
  if (status === 401 || status === 422) return false;
  return status === 429 || status === 529 || status >= 500;
}
