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
      'Why is even the equator so cold in this model?',
      'Why are the poles frozen?',
      'What\'s making water temperatures more stable than land temperatures?',
    ],
  },
  {
    button: 'Stir the air',
    description: 'A blanket of air traps heat...',
    chatSuggestions: [
      'Why doesn\'t anything stop the tropics from overheating?',
      'Why is the planet so uneven?',
      'How much warming does the atmosphere add?',
    ],
  },
  {
    button: 'Grow rainforests and deserts',
    description: 'Wind and eddies carry heat...',
    chatSuggestions: [
      'What\'s the difference between winds and eddies?',
      'Why are the winds going east to west towards the equator?',
      'Why are the tropical rainforests still so dry?',
    ],
  },
  {
    button: 'Add currents and life',
    description: 'Hadley cells create deserts and trade winds...',
    chatSuggestions: [
      'What are Hadley cells?',
      'Why do deserts form at 30\u00b0?',
      'Why does the wind flip direction at 30\u00b0?',
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
