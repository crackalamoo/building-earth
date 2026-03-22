import { writable } from 'svelte/store';

export type Stage = 0 | 1 | 2 | 3 | 4;

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
    button: 'Add water & life',
    description: 'Wind carries heat from equator to poles...',
    chatSuggestions: [
      'How does wind redistribute heat?',
      'Why do westerlies form?',
      'What are Hadley cells?',
    ],
  },
  {
    button: null,
    description: 'Water evaporates, clouds form, rain falls...',
    chatSuggestions: [],  // use default dynamic suggestions at stage 4
  },
];

/** Stage names sent to the chat API for LLM context */
export const STAGE_NAMES = [
  'Primordial',
  'Radiation Only',
  'Atmosphere & Greenhouse',
  'Diffusion & Wind',
  'Full Model',
];
