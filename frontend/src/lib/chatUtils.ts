import { STAGES } from './onboardingState';

export type MsgPart =
  | { type: 'text'; content: string }
  | { type: 'tools'; content: string; fields: string[]; pendingCount?: number };

export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  parts?: MsgPart[];
  thinking?: string;
};

export function computeSuggestions(
  lat: number,
  ocean: { isOcean: boolean; sstC: number | null },
  elevation: number | null,
  cycleTemps: number[],
  cyclePrecip: number[],
  wind: { speed: number; dir: number },
  currentMonthIdx: number,
  _tempC: number,
  stage: number = 5,
  obsTemps: (number | null)[] = [],
  obsPrecips: (number | null)[] = [],
): string[] {
  const candidates: string[] = [];

  // Stage-specific suggestions (stages 1-3)
  if (stage <= 4) {
    const suggestions = STAGES[stage]?.chatSuggestions;
    return suggestions?.length ? suggestions : candidates.slice(0, 3);
  }

  const hasObs = obsTemps.some(v => v !== null) && obsPrecips.some(v => v !== null);

  const tempMax = Math.max(...cycleTemps);
  const tempMin = Math.min(...cycleTemps);
  const seasonalSwing = tempMax - tempMin;
  const avgMonthlyPrecip = cyclePrecip.reduce((a, b) => a + b, 0) / 12;
  const currentPrecip = cyclePrecip[currentMonthIdx];
  const coldestMonth = cycleTemps.indexOf(tempMin);
  const warmestMonth = cycleTemps.indexOf(tempMax);

  // Obs-derived stats (only used when obs data is available)
  const obsTempsValid = obsTemps.filter((v): v is number => v !== null);
  const obsPrecipsValid = obsPrecips.filter((v): v is number => v !== null);
  const obsAvgPrecip = obsPrecipsValid.length
    ? obsPrecipsValid.reduce((a, b) => a + b, 0) / obsPrecipsValid.length
    : null;
  const obsCurrentPrecip = obsPrecips[currentMonthIdx] ?? null;
  const obsSwing = obsTempsValid.length >= 2
    ? Math.max(...obsTempsValid) - Math.min(...obsTempsValid)
    : null;
  const obsColdestMonth = obsTempsValid.length === 12
    ? obsTemps.indexOf(Math.min(...obsTempsValid))
    : null;
  const obsWarmestMonth = obsTempsValid.length === 12
    ? obsTemps.indexOf(Math.max(...obsTempsValid))
    : null;

  // Helper: only add suggestion if obs agrees (or no obs available)
  function agree(simCondition: boolean, obsCondition: boolean | null): boolean {
    if (!hasObs || obsCondition === null) return simCondition;
    return simCondition && obsCondition;
  }

  if (elevation !== null && elevation > 1500)
    candidates.push('Why is it cooler at this elevation?');

  if (ocean.isOcean)
    candidates.push('How does the ocean shape the climate here?');

  if (agree(avgMonthlyPrecip < 10, obsAvgPrecip !== null ? obsAvgPrecip < 10 : null))
    candidates.push('Why is it so dry here?');

  if (agree(currentPrecip > 100, obsCurrentPrecip !== null ? obsCurrentPrecip > 100 : null))
    candidates.push('Why does it rain so much here?');

  if (agree(seasonalSwing > 25, obsSwing !== null ? obsSwing > 25 : null))
    candidates.push('Why are the seasons so extreme here?');
  else if (agree(seasonalSwing < 8, obsSwing !== null ? obsSwing < 8 : null))
    candidates.push('Why is the temperature so stable year-round?');

  const monthNames = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const sh = lat < 0;
  const boringWarmest = sh ? [0, 1, 11] : [5, 6, 7];
  const boringColdest = sh ? [5, 6, 7] : [0, 1, 11];
  if (!boringWarmest.includes(warmestMonth) && (obsWarmestMonth === null || (!boringWarmest.includes(obsWarmestMonth) && obsWarmestMonth === warmestMonth)))
    candidates.push(`Why is ${monthNames[warmestMonth]} the warmest time of year here?`);
  if (!boringColdest.includes(coldestMonth) && (obsColdestMonth === null || (!boringColdest.includes(obsColdestMonth) && obsColdestMonth === coldestMonth)))
    candidates.push(`Why is ${monthNames[coldestMonth]} the coldest time of year here?`);

  if (wind.speed > 8)
    candidates.push('Why is it so windy here?');

  candidates.push('What makes this place\'s climate unique?');

  return candidates.slice(0, 3);
}

export type StreamChatCallbacks = {
  onPart: (parts: MsgPart[]) => void;
  onContent: (content: string) => void;
  // Reasoning summary delta. `reset` is true when this delta starts a fresh
  // round of thinking after a tool call — the consumer should clear any
  // accumulated thinking text before appending.
  onThinking: (delta: string, reset: boolean) => void;
  onError: (msg: string) => void;
  onDone: () => void;
};

export async function streamChat(
  apiBase: string,
  payload: {
    lat: number;
    lon: number;
    prevLat: number | null;
    prevLon: number | null;
    imperial: boolean;
    messages: { role: string; content: string }[];
    stage?: number;
  },
  signal: AbortSignal,
  callbacks: StreamChatCallbacks
): Promise<void> {
  const { onPart, onContent, onThinking, onError, onDone } = callbacks;
  const currentParts: MsgPart[] = [];
  // Set when a tool/text event arrives, cleared on the next thinking delta.
  // Tells the consumer that the next thinking delta is a fresh round and
  // any previously displayed thinking should be cleared first.
  let thinkingStale = false;

  const res = await fetch(`${apiBase}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();
  if (!reader) throw new Error('No response body');

  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') { onDone(); return; }
      try {
        const parsed = JSON.parse(data);
        if (parsed.error) {
          const last = currentParts[currentParts.length - 1];
          if (last?.type === 'text') last.content += parsed.error;
          else currentParts.push({ type: 'text', content: parsed.error });
          onContent(parsed.error);
          thinkingStale = true;
        } else if (parsed.tool) {
          const last = currentParts[currentParts.length - 1];
          const isPending = parsed.pending === true;
          if (last?.type === 'tools') {
            if (isPending) {
              last.fields = [...last.fields, parsed.tool];
              last.pendingCount = (last.pendingCount ?? 0) + 1;
            } else {
              // First real label arriving — drop any pending placeholders
              // accumulated for this round before appending.
              const pending = last.pendingCount ?? 0;
              const kept = pending > 0 ? last.fields.slice(0, last.fields.length - pending) : last.fields;
              last.fields = [...kept, parsed.tool];
              last.pendingCount = 0;
            }
          } else {
            currentParts.push({
              type: 'tools',
              content: '',
              fields: [parsed.tool],
              pendingCount: isPending ? 1 : 0,
            });
          }
          thinkingStale = true;
        } else if (parsed.text) {
          const last = currentParts[currentParts.length - 1];
          if (last?.type === 'text') last.content += parsed.text;
          else currentParts.push({ type: 'text', content: parsed.text });
          onContent(parsed.text);
          thinkingStale = true;
        } else if (parsed.thinking) {
          // Reasoning summary delta — drive the live "thinking" UI directly,
          // not via parts. The consumer clears it once any text part starts.
          onThinking(parsed.thinking, thinkingStale);
          thinkingStale = false;
          continue;
        } else {
          continue;
        }
        onPart([...currentParts]);
      } catch { /* skip malformed chunks */ }
    }
  }
  onDone();
}
