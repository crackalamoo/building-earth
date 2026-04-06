import { writable } from 'svelte/store';

export type Stage = 0 | 1 | 2 | 3 | 4 | 5;

export const currentStage = writable<Stage>(0);
export const stageLoading = writable(false);

export interface StageInfo {
  button: string | null;
  description: string;
  chatSuggestions: string[];
}

export const STAGES: StageInfo[] = [
  {
    button: 'Let there be light',
    description: '',
    chatSuggestions: [],
  },
  {
    button: 'Add an atmosphere',
    description: 'Raw sunlight on bare rock...',
    chatSuggestions: [
      'Why is the equator so hot?',
      'Why are the poles frozen?',
      'What\'s missing from this model?',
    ],
  },
  {
    button: 'Stir the air',
    description: 'A blanket of air traps heat...',
    chatSuggestions: [
      'How does the greenhouse effect warm the poles?',
      'Why is it still unrealistic?',
      'How much warming does the atmosphere add?',
    ],
  },
  {
    button: 'Grow rainforests and deserts',
    description: 'Wind and eddies carry heat...',
    chatSuggestions: [
      'How does wind redistribute heat?',
      'Why do westerlies form?',
      'Where does it rain?',
    ],
  },
  {
    button: 'Add currents and life',
    description: 'Hadley cells create deserts and trade winds...',
    chatSuggestions: [
      'What are Hadley cells?',
      'Why do deserts form at 30\u00b0?',
      'How do trade winds work?',
    ],
  },
  {
    button: null,
    description: 'Ocean currents complete the machine...',
    chatSuggestions: [],  // use default dynamic suggestions at stage 5
  },
];

/** Stage names sent to the chat API for LLM context */
export const STAGE_NAMES = [
  'Primordial',
  'Radiation Only',
  'Atmosphere & Greenhouse',
  'Wind & Diffusion',
  'Hadley Circulation',
  'Full Model',
];
