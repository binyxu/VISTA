/**
 * Lazy singleton embedding model using @xenova/transformers.
 * Falls back gracefully to null if the package is not installed.
 *
 * Model: Xenova/all-MiniLM-L6-v2
 *   - 384-dim, ~23MB quantized, zero API cost, runs locally in Node.js
 *   - First call downloads the model; subsequent calls use the cache.
 */

let _embedFn = null;
let _loadPromise = null;
let _unavailable = false;

async function _load() {
  try {
    const { pipeline } = await import("@xenova/transformers");
    const pipe = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
      { quantized: true }
    );
    return async (text) => {
      const out = await pipe(String(text).slice(0, 4096), {
        pooling: "mean",
        normalize: true,
      });
      // Return plain Array for JSON serialization
      return Array.from(out.data);
    };
  } catch {
    _unavailable = true;
    return null;
  }
}

/**
 * Returns the embedding function, or null if @xenova/transformers is not available.
 * @returns {Promise<((text: string) => Promise<number[]>) | null>}
 */
export async function getEmbedFn() {
  if (_unavailable) return null;
  if (_embedFn) return _embedFn;
  if (!_loadPromise) _loadPromise = _load();
  _embedFn = await _loadPromise;
  return _embedFn;
}

/**
 * Cosine similarity between two normalized vectors (dot product suffices).
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
export function cosine(a, b) {
  let dot = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) dot += a[i] * b[i];
  return dot;
}
