class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // OPTIONAL: Buffer slightly to reduce network packet count (e.g. 2048 samples)
    // For lowest latency, we send every chunk (128 samples) immediately.
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      // Input comes in as Float32Array (channel 0)
      // We copy it to ensure the memory isn't detached before sending
      const outputData = new Float32Array(input[0]);
      
      // Post as binary ArrayBuffer so the backend can np.frombuffer(message, dtype=np.float32)
      this.port.postMessage(outputData.buffer, [outputData.buffer]);
    }
    return true; // Keep processor alive
  }
}

registerProcessor('audio-processor', AudioProcessor);