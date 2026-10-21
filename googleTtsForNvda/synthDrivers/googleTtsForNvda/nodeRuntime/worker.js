// node_worker.js — the long-lived Node process that replaces headless Chromium.
//
// Load order matters:
//   1. node_shim.js        — window/self, relative-URL fetch, in-memory indexedDB
//   2. ../web/bridgeHarness.js — shared verbatim with the browser path
//   3. engine_boot.js      — generated: bindings_main.js + offscreen_compiled.js
//
// Protocol — one JSON object per line. stdout carries protocol JSON only; all
// logging goes to stderr.
//
//   stdin : {"id":<int>,"op":"speak"|"preload"|"stop"|"ready","payload":{...}}
//           {"op":"stop"}                       (fire-and-forget, no id)
//   stdout: {"method":"Runtime.ready","harnessReady":true}
//           {"method":"Runtime.bindingCalled","params":{"name":...,"payload":"<json>"}}
//           {"id":<int>,"ok":true,"result":{...}} | {"id":<int>,"ok":false,"error":"..."}
"use strict";

const path = require("node:path");

const BINDING_NAME = process.env.GTTS_BINDING_NAME || "googleTtsForNvdaBridge";
const WEB_DIR = process.env.GTTS_WEB_DIR || "";
const MAIN_ENTRY = process.env.GTTS_MAIN_ENTRY || "";
const PREVIOUS_SPEAK_SETTLE_TIMEOUT_MS = Number(process.env.GTTS_SPEAK_SETTLE_TIMEOUT_MS) || 1000;
const DRAIN_IDLE_MS = Number(process.env.GTTS_DRAIN_IDLE_MS) || 250;

function log(...args) {
	console.error("[nodeRuntime]", ...args);
}

function writeLine(object) {
	try {
		process.stdout.write(JSON.stringify(object) + "\n");
	} catch (error) {
		log("failed to serialise protocol message:", String(error));
	}
}

function fail(id, error) {
	writeLine({ id, ok: false, error: (error && error.message) || String(error) });
}

// Audio accounting, so Python can prove whether a truncated utterance lost samples
// in transport or never had them synthesised.
const emitted = { packets: 0, samples: 0 };

globalThis[BINDING_NAME] = function runtimeBridge(payloadJson) {
	const text = String(payloadJson);
	try {
		const message = JSON.parse(text);
		if (message && message.type === "audio") {
			emitted.packets += 1;
			// bridgeHarness.js sends {type,sampleRate,data} with data as base64 s16 PCM
			// and no explicit sample count, so derive it from the payload length.
			const encoded = String(message.data || "");
			const padding = encoded.length - encoded.replace(/=+$/, "").length;
			const byteCount = Math.floor((encoded.length * 3) / 4) - padding;
			emitted.samples += Math.max(0, Math.floor(byteCount / 2));
		}
	} catch (error) {
		// Never let accounting break the protocol stream.
	}
	writeLine({
		method: "Runtime.bindingCalled",
		params: { name: BINDING_NAME, payload: text },
	});
};

let loadError = null;
try {
	if (!WEB_DIR || !MAIN_ENTRY) {
		throw new Error("GTTS_WEB_DIR and GTTS_MAIN_ENTRY must be set");
	}
	require(path.join(__dirname, "shim.js"));
	// Must be set before bridgeHarness.js reads it into a const. See the harness
	// comment on synthesisFinishedIdleMs: an idle gap in the engine's audio pump is
	// the only completion criterion once an utterance has produced sound, so a host
	// that shares one event loop with the WASM pthread pool needs more slack than a
	// dedicated Chromium renderer does.
	globalThis.googleTtsForNvdaDrainIdleMs = DRAIN_IDLE_MS;
	require(path.join(WEB_DIR, "bridgeHarness.js"));
	require(MAIN_ENTRY);
} catch (error) {
	loadError = error;
	log("engine load failed:", (error && error.stack) || String(error));
}

const ready = !loadError && typeof globalThis.googleTtsForNvdaSpeak === "function";
if (loadError) {
	writeLine({ method: "Runtime.loadError", error: (loadError && loadError.message) || String(loadError) });
} else {
	writeLine({ method: "Runtime.ready", harnessReady: ready });
	log("worker ready; harness functions:", typeof globalThis.googleTtsForNvdaSpeak);
}

// bridgeHarness.js's googleTtsForNvdaSpeak() success tail fences every call with
// sessionToken EXCEPT its final `await stopActiveSynthesis()`, which unconditionally
// bumps the token, resets the audio queue and stops the engine. A superseded
// invocation unwinding after the next session began would therefore truncate the NEW
// utterance. Serialize so two invocations can never overlap.
let activeSpeak = null;

function settleWithBound(promise, milliseconds) {
	let timer = null;
	const timeout = new Promise((resolve) => {
		timer = setTimeout(() => resolve("timeout"), milliseconds);
	});
	return Promise.race([
		Promise.resolve(promise).then(
			() => "settled",
			() => "settled",
		),
		timeout,
	]).then((outcome) => {
		if (timer !== null) clearTimeout(timer);
		return outcome;
	});
}

async function speakExclusive(payload) {
	const previous = activeSpeak;
	if (previous) {
		log("speak arrived while a previous utterance was still in flight; settling it first");
		try {
			if (typeof globalThis.googleTtsForNvdaStop === "function") {
				await globalThis.googleTtsForNvdaStop();
			}
		} catch (error) {
			log("stop before re-speak failed:", String(error));
		}
		const outcome = await settleWithBound(previous, PREVIOUS_SPEAK_SETTLE_TIMEOUT_MS);
		if (outcome === "timeout") {
			log("previous utterance did not settle within", PREVIOUS_SPEAK_SETTLE_TIMEOUT_MS, "ms; starting anyway");
		}
	}
	emitted.packets = 0;
	emitted.samples = 0;
	const current = globalThis.googleTtsForNvdaSpeak(payload);
	activeSpeak = current;
	try {
		const result = await current;
		if (result && typeof result === "object") {
			result.jsAudioPackets = emitted.packets;
			result.jsAudioSamples = emitted.samples;
		}
		return result;
	} finally {
		if (activeSpeak === current) {
			activeSpeak = null;
		}
	}
}

async function runOp(message) {
	const op = message.op;
	const payload = message.payload || {};
	if (!ready) {
		throw new Error("Bundled speech runtime failed to load the WASM TTS engine.");
	}
	if (op === "ready") {
		const isReady =
			typeof globalThis.googleTtsForNvdaReady === "function" && globalThis.googleTtsForNvdaReady() === true;
		return { ready: isReady };
	}
	if (op === "stop") {
		if (typeof globalThis.googleTtsForNvdaStop === "function") {
			await globalThis.googleTtsForNvdaStop();
		}
		return { stopped: true };
	}
	if (op === "preload") {
		return await globalThis.googleTtsForNvdaPreload(payload);
	}
	if (op === "speak") {
		return await speakExclusive(payload);
	}
	throw new Error("Unknown operation: " + String(op));
}

const inflight = new Map();

function dispatch(line) {
	let message;
	try {
		message = JSON.parse(line);
	} catch (error) {
		log("ignoring malformed stdin line:", line.slice(0, 120));
		return;
	}
	const id = message.id;
	if (id === undefined || id === null) {
		// Fire-and-forget command from send_command_async(), which is how
		// send_fast_stop() cancels speech. These must not be dropped.
		runOp(message).catch((error) => {
			log("fire-and-forget command failed:", (error && error.message) || String(error));
		});
		return;
	}
	if (inflight.has(id)) {
		log("ignoring duplicate request id", id);
		return;
	}
	inflight.set(id, true);
	runOp(message).then(
		(result) => {
			inflight.delete(id);
			writeLine({ id, ok: true, result: result === undefined ? null : result });
		},
		(error) => {
			inflight.delete(id);
			fail(id, error);
		},
	);
}

let pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
	pending += chunk;
	let newline = pending.indexOf("\n");
	while (newline >= 0) {
		const line = pending.slice(0, newline).trim();
		pending = pending.slice(newline + 1);
		newline = pending.indexOf("\n");
		if (line) {
			dispatch(line);
		}
	}
});
process.stdin.on("end", () => process.exit(0));
process.stdin.on("error", () => process.exit(0));
process.on("uncaughtException", (error) => {
	log("uncaught exception:", (error && error.stack) || String(error));
	writeLine({ method: "Runtime.crashed", error: (error && error.message) || String(error) });
	process.exit(1);
});
