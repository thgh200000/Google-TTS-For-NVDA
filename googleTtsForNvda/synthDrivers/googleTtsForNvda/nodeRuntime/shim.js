// node_shim.js — the ONLY host-API shim Node needs.
//
// Compare with the Node shim: gaps 1 and 2 are gone.
//
//   Gap 1 (worker messages never reach globalThis.onmessage) does not exist here.
//     Emscripten's glue installs its own parentPort bridge whenever
//     `globalThis.postMessage` is absent, and Node does not define a global
//     postMessage in a worker_threads worker -- so the bridge is installed.
//     Node defines globalThis.postMessage, which made Emscripten skip it.
//
//   Gap 2 (pthread workers never self-boot; wrong Worker class) does not exist
//     either. bindings_main.js ends with
//         var isNode = process?.versions?.node && process?.type != 'renderer';
//         if (isNode) isPthread = require('node:worker_threads').workerData === 'em-pthread';
//         isPthread && loadWasmTtsBindings();
//     which is exactly right under Node, and Emscripten itself sets
//     globalThis.Worker = worker_threads.Worker under ENVIRONMENT_IS_NODE, so the
//     main-thread path gets a Worker with .on()/.unref().
//
//   Gap 3 remains: no indexedDB (IDBFS asserts) and fetch() cannot resolve the
//     relative URLs the engine uses. Both are handled below.
//
// Consequence: pthread workers can run the REAL, unmodified bindings_main.js
// directly. There is no generated worker entry and no Worker redirection.
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const ENGINE_DIR = process.env.GTTS_ENGINE_DIR || "";
const VOICE_DIR = process.env.GTTS_VOICE_DIR || "";
const VOICES_JSON = process.env.GTTS_VOICES_JSON || "";
const DEBUG = process.env.GTTS_DEBUG === "1";

function debug(...args) {
	if (DEBUG) console.error("[nodeRuntime]", ...args);
}

// The engine and the harness are browser scripts; they need a global `window`.
globalThis.window = globalThis.window || globalThis;
globalThis.self = globalThis.self || globalThis;

// --- Gap 3a: relative-URL fetch, replacing bridge.py's local HTTP server. -----
// Mirrors _BridgeRequestHandler.translate_path(): "voices.json" resolves to the
// generated runtime catalog and any "*.zvoice" resolves inside the voice package
// directory. Traversal is rejected the way _safe_join() rejects it.
function safeJoin(root, relative) {
	const resolvedRoot = path.resolve(root);
	const target = path.resolve(resolvedRoot, decodeURIComponent(String(relative).replace(/^[\\/]+/, "")));
	if (target !== resolvedRoot && !target.startsWith(resolvedRoot + path.sep)) {
		return null;
	}
	return target;
}

function resolveLocalUrl(url) {
	const route = String(url).split("?")[0].split("#")[0];
	if (route === "/" || route === "") {
		return null;
	}
	if (route.replace(/^\/+/, "") === "voices.json") {
		return VOICES_JSON || null;
	}
	if (route.startsWith("/engine/")) {
		return ENGINE_DIR ? safeJoin(ENGINE_DIR, route.slice("/engine/".length)) : null;
	}
	if (route.endsWith(".zvoice")) {
		return VOICE_DIR ? safeJoin(VOICE_DIR, route.replace(/^\/+/, "")) : null;
	}
	return null;
}

(function installFetchShim() {
	if (globalThis.__gttsFetchShim) return;
	globalThis.__gttsFetchShim = true;
	const nativeFetch = globalThis.fetch;
	globalThis.fetch = async function patchedFetch(input, init) {
		const url = typeof input === "string" ? input : String((input && input.url) || input);
		if (/^[a-z][a-z0-9+.-]*:\/\//i.test(url)) {
			return nativeFetch(input, init);
		}
		const file = resolveLocalUrl(url);
		debug("fetch", JSON.stringify(url), "->", file || "NOT FOUND");
		if (!file || !fs.existsSync(file)) {
			return new Response("not found", { status: 404, statusText: "Not Found" });
		}
		return new Response(fs.readFileSync(file), {
			status: 200,
			statusText: "OK",
			headers: { "Content-Type": "application/octet-stream" },
		});
	};
})();

// --- Gap 3b: minimal in-memory indexedDB ------------------------------------
// IDBFS.mount() delegates to MEMFS, so mounting is harmless; only FS.syncfs()
// touches indexedDB, where it asserts. The cache is redundant here -- voice
// packages live on disk and are SHA-256 verified by voice_store.py, and reach the
// engine through the fetch shim, never through /voices. An empty database is
// therefore correct, and IDBFS.reconcile() short-circuits on `if (!total)`.
(function installIndexedDbStub() {
	if (typeof globalThis.indexedDB !== "undefined") return;

	class FakeRequest {
		constructor() {
			this.result = undefined;
			this.error = null;
			this.transaction = null;
			this.onsuccess = null;
			this.onerror = null;
			this.onupgradeneeded = null;
		}
		_fire(handler, target) {
			if (typeof handler !== "function") return;
			try {
				handler({ target: target || this, type: "success" });
			} catch (_) {}
		}
	}
	class FakeIndex {
		constructor(name) {
			this.name = name || "timestamp";
		}
		openKeyCursor() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess, { result: null }), 0);
			return request;
		}
		openCursor() {
			return this.openKeyCursor();
		}
		get() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess, { result: undefined }), 0);
			return request;
		}
	}
	class FakeObjectStore {
		constructor(name) {
			this.name = name;
			this.keyPath = null;
			this.indexNames = { contains: () => false };
		}
		index(name) {
			return new FakeIndex(name);
		}
		createIndex(name) {
			return new FakeIndex(name);
		}
		_ack() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess), 0);
			return request;
		}
		put() {
			return this._ack();
		}
		add() {
			return this._ack();
		}
		delete() {
			return this._ack();
		}
		clear() {
			return this._ack();
		}
		get() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess, { result: undefined }), 0);
			return request;
		}
		openCursor() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess, { result: null }), 0);
			return request;
		}
		count() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess, { result: 0 }), 0);
			return request;
		}
	}
	class FakeTransaction {
		constructor(stores) {
			this._stores = stores || {};
			this.onerror = null;
			this.onabort = null;
			this.oncomplete = null;
		}
		objectStore(name) {
			if (!this._stores[name]) this._stores[name] = new FakeObjectStore(name);
			return this._stores[name];
		}
		_complete() {
			if (typeof this.oncomplete === "function") {
				try {
					this.oncomplete({ target: this });
				} catch (_) {}
			}
		}
	}
	class FakeDatabase {
		constructor() {
			this._stores = {};
			this.onerror = null;
			this.onversionchange = null;
			this.objectStoreNames = { contains: (name) => !!this._stores[name] };
		}
		createObjectStore(name) {
			this._stores[name] = new FakeObjectStore(name);
			return this._stores[name];
		}
		deleteObjectStore(name) {
			delete this._stores[name];
		}
		transaction(names) {
			const requested = Array.isArray(names) ? names : [names];
			const transaction = new FakeTransaction(this._stores);
			for (const name of requested) {
				if (!this._stores[name]) this._stores[name] = new FakeObjectStore(name);
			}
			setTimeout(() => transaction._complete(), 0);
			return transaction;
		}
		close() {}
	}

	globalThis.indexedDB = {
		open() {
			const request = new FakeRequest();
			const database = new FakeDatabase();
			request.result = database;
			setTimeout(() => {
				const transaction = new FakeTransaction(database._stores);
				request.transaction = transaction;
				request._fire(request.onupgradeneeded, { result: database, transaction });
				request._fire(request.onsuccess, { result: database });
				transaction._complete();
			}, 0);
			return request;
		},
		deleteDatabase() {
			const request = new FakeRequest();
			setTimeout(() => request._fire(request.onsuccess), 0);
			return request;
		},
		databases: async () => [],
		cmp: () => 0,
	};
	globalThis.IDBKeyRange = { only: (value) => value, bound: () => null };
	debug("installed in-memory indexedDB stub");
})();

module.exports = { ENGINE_DIR, VOICE_DIR, VOICES_JSON, debug, resolveLocalUrl, safeJoin };
