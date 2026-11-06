export async function api(url, { timeout: timeoutMs = 45000, signal: caller, format, ...options } = {}) {
  const timeout = AbortSignal.timeout(timeoutMs);
  const signal = caller ? AbortSignal.any([caller, timeout]) : timeout;
  try {
    const response = await fetch(url, { ...options, signal });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.message || `The service returned an error (${response.status}).`);
    }
    return format === "blob" ? await response.blob() : await response.json();
  } catch (error) {
    if (timeout.aborted) throw new Error("The request timed out. Please try again.");
    if (error instanceof TypeError) throw new Error("Could not reach VisionGuard. Please try again.");
    throw error;
  }
}
