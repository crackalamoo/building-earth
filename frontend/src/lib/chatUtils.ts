import { STAGES } from './onboardingState';

export type MsgPart =
  | { type: 'text'; content: string }
  | { type: 'tools'; content: string; fields: string[] };

export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  parts?: MsgPart[];
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
  stage: number = 4,
): string[] {
  const candidates: string[] = [];

  // Stage-specific suggestions (stages 1-3)
  if (stage <= 3) {
    const suggestions = STAGES[stage]?.chatSuggestions;
    return suggestions?.length ? suggestions : candidates.slice(0, 3);
  }

  const tempMax = Math.max(...cycleTemps);
  const tempMin = Math.min(...cycleTemps);
  const seasonalSwing = tempMax - tempMin;
  const avgMonthlyPrecip = cyclePrecip.reduce((a, b) => a + b, 0) / 12;
  const currentPrecip = cyclePrecip[currentMonthIdx];
  const coldestMonth = cycleTemps.indexOf(tempMin);
  const warmestMonth = cycleTemps.indexOf(tempMax);

  if (elevation !== null && elevation > 1500)
    candidates.push('Why is it cooler at this elevation?');

  if (ocean.isOcean) {
    candidates.push('Why is the sea temperature different from the air?');
  }

  if (avgMonthlyPrecip < 10)
    candidates.push('Why is it so dry here?');

  if (currentPrecip > 100)
    candidates.push('What drives the heavy rainfall here?');

  if (seasonalSwing > 25)
    candidates.push('Why is the seasonal swing so large here?');
  else if (seasonalSwing < 8)
    candidates.push('Why is the temperature so stable year-round?');

  const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const sh = lat < 0;
  const boringWarmest = sh ? [0, 1, 11] : [5, 6, 7];
  const boringColdest = sh ? [5, 6, 7] : [0, 11];
  if (!boringWarmest.includes(warmestMonth))
    candidates.push(`Why is ${monthNames[warmestMonth]} the warmest time of year here?`);
  if (!boringColdest.includes(coldestMonth))
    candidates.push(`Why is ${monthNames[coldestMonth]} the coldest time of year here?`);

  if (wind.speed > 8)
    candidates.push('What drives the strong winds here?');

  candidates.push('What affects the climate here?');

  return candidates.slice(0, 3);
}

export type StreamChatCallbacks = {
  onPart: (parts: MsgPart[]) => void;
  onContent: (content: string) => void;
  onError: (msg: string) => void;
  onDone: () => void;
};

export async function streamChat(
  apiBase: string,
  payload: {
    lat: number;
    lon: number;
    month: number;
    prevLat: number | null;
    prevLon: number | null;
    prevMonth: number | null;
    imperial: boolean;
    messages: { role: string; content: string }[];
    stage?: number;
  },
  signal: AbortSignal,
  callbacks: StreamChatCallbacks
): Promise<void> {
  const { onPart, onContent, onError, onDone } = callbacks;
  const currentParts: MsgPart[] = [];

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
        } else if (parsed.tool) {
          const last = currentParts[currentParts.length - 1];
          if (last?.type === 'tools') last.fields = [...last.fields, parsed.tool];
          else currentParts.push({ type: 'tools', content: '', fields: [parsed.tool] });
        } else if (parsed.text) {
          const last = currentParts[currentParts.length - 1];
          if (last?.type === 'text') last.content += parsed.text;
          else currentParts.push({ type: 'text', content: parsed.text });
          onContent(parsed.text);
        } else {
          continue;
        }
        onPart([...currentParts]);
      } catch { /* skip malformed chunks */ }
    }
  }
  onDone();
}
