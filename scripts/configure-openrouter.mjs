#!/usr/bin/env node
/**
 * Point jev at OpenRouter in Claude Code's user settings.
 *
 * The Claude Code desktop app does not render a plugin's `userConfig` fields, and
 * it does not inherit `export`s from a terminal. The `env` block of
 * ~/.claude/settings.json is applied to every session, the hooks and the MCP
 * server alike, so that is where the provider and key go.
 *
 *   export OPENROUTER_API_KEY=sk-or-...
 *   node scripts/configure-openrouter.mjs            # or: --provider typesafe
 *
 * The key is read from the environment (never from argv, so it stays out of shell
 * history and `ps`), written to settings.json with mode 600, and never printed.
 * The previous file is kept as settings.json.bak-jev.
 */

import { chmodSync, copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const provider = process.argv.includes("--provider")
  ? process.argv[process.argv.indexOf("--provider") + 1]
  : "openrouter";
const keyVar = { openrouter: "OPENROUTER_API_KEY", typesafe: "TYPESAFE_API_KEY" }[provider ?? ""];
if (keyVar === undefined) {
  console.error("--provider must be openrouter or typesafe");
  process.exit(2);
}

const key = (process.env[keyVar] ?? "").trim();
if (key === "") {
  console.error(`${keyVar} is not set in this shell. Run: export ${keyVar}=... and try again.`);
  process.exit(1);
}

const path = join(process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude"), "settings.json");
let settings = {};
if (existsSync(path)) {
  copyFileSync(path, `${path}.bak-jev`);
  settings = JSON.parse(readFileSync(path, "utf8"));
}
settings.env = { ...(settings.env ?? {}), JEV_PROVIDER: provider, [keyVar]: key };

writeFileSync(path, `${JSON.stringify(settings, null, 2)}\n`, { mode: 0o600 });
chmodSync(path, 0o600);
console.log(`jev: JEV_PROVIDER=${provider} and ${keyVar} written to ${path} (key not shown).`);
console.log("Restart Claude Code, then run /jev:daemon restart and /jev:status.");
