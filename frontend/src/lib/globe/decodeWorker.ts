/**
 * Web Worker for decoding binary climate data off the main thread.
 * Receives: { buffer: ArrayBuffer, manifest: Manifest }
 * Returns: { fields: Record<string, { data: buffer, shape }>, temperatureNested: number[][][] }
 */

interface ManifestField {
  name: string;
  shape: number[];
  dtype: 'float16' | 'uint8' | 'float32';
  offset: number;
  bytes: number;
}

interface Manifest {
  fields: ManifestField[];
}

function decodeFloat16(buffer: ArrayBuffer, byteOffset: number, count: number): Float32Array {
  const u16 = new Uint16Array(buffer, byteOffset, count);
  const f32 = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const h = u16[i];
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;

    let value: number;
    if (exponent === 0) {
      value = (mantissa / 1024) * Math.pow(2, -14);
    } else if (exponent === 31) {
      value = mantissa === 0 ? Infinity : NaN;
    } else {
      value = (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
    }

    f32[i] = sign ? -value : value;
  }

  return f32;
}

function decodeField(buffer: ArrayBuffer, field: ManifestField): Float32Array | Uint8Array {
  if (field.dtype === 'uint8') {
    // Copy out of the shared buffer so it can be transferred independently
    const src = new Uint8Array(buffer, field.offset, field.bytes);
    const copy = new Uint8Array(field.bytes);
    copy.set(src);
    return copy;
  }
  if (field.dtype === 'float16') {
    const count = field.bytes / 2;
    return decodeFloat16(buffer, field.offset, count);
  }
  // float32 — copy out of shared buffer
  const src = new Float32Array(buffer, field.offset, field.bytes / 4);
  const copy = new Float32Array(src.length);
  copy.set(src);
  return copy;
}

self.onmessage = (e: MessageEvent<{ buffer: ArrayBuffer; manifest: Manifest }>) => {
  const { buffer, manifest } = e.data;

  const fields: Record<string, { data: Float32Array | Uint8Array; shape: number[] }> = {};
  const transferables: ArrayBuffer[] = [];

  for (const field of manifest.fields) {
    let data = decodeField(buffer, field);

    // Decode uint8-quantized temperature_2m back to °C
    if (field.name === 'temperature_2m' && field.dtype === 'uint8') {
      const u8 = data as Uint8Array;
      const f32 = new Float32Array(u8.length);
      for (let i = 0; i < u8.length; i++) f32[i] = u8[i] * (120 / 255) - 60;
      data = f32;
    }

    fields[field.name] = { data, shape: field.shape };
    transferables.push(data.buffer);
  }

  // Only transfer decoded typed arrays — no nested JS arrays (structured clone is too expensive)
  self.postMessage({ fields }, transferables as any);
};
