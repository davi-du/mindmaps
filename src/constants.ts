export const debug = process.env.NODE_ENV === 'development'
export const useMagic = true // enable OpenAI? // deprecated
export const useSessionStorage = true // TODO ?

export const useSessionStorageHandle = '__reactFlowGraphologue__'
export const useTokenDataTransferHandle = '__reactFlowGraphologueToken__'
export const useSessionStorageNotesHandle = '__reactFlowGraphologueNotes__'

export const viewFittingPadding = 0.05
export const transitionDuration = 1500
export const viewFittingOptions = {
  padding: viewFittingPadding,
  duration: transitionDuration,
}
export const hideEdgeTextZoomLevel = 0.6

export const timeMachineMaxSize = 50
export const contentEditingTimeout = 500
export const slowInteractionWaitTimeout = 100

export const hardcodedNodeSize = {
  width: 160,
  height: 43,
  magicWidth: 320,
  magicHeight: 160,
}
export const nodeGap = 30 // ?
export const nodePosAdjustStep = 50

export const styles = {
  edgeColorStrokeDefault: '#aaaaaa',
  edgeColorStrokeSelected: '#ff4d00',
  edgeColorStrokeExplained: '#57068c',
  edgeColorLabelDefault: '#666666',
  edgeWidth: 2,
  edgeMarkerSize: 12,
  edgeDashLineArray: '5, 7',
  nodeColorDefaultWhite: '#ffffff',
  nodeColorDefaultGrey: '#999999',
}

export const terminalColor = {
  Reset: "\x1b[0m",
  Bright: "\x1b[1m",
  FgRed: "\x1b[31m",
  FgGreen: "\x1b[32m",
  FgYellow: "\x1b[33m",
  FgBlue: "\x1b[34m"
}

/*
export const terms = {
  'codestral-2405': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
  'gpt-4': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
  'codestral-latest': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
  wiki: 'Wikidata', // deprecated
}
*/

export const terms = {
  'fireworks-llama': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
  wiki: 'Wikidata', // deprecated
}

export const magicNodeVerifyPaperCountDefault = 3

/* -------------------------------------------------------------------------- */

export const userProvidedAPIKey = {
  current: process.env.REACT_APP_FIREWORKS_API_KEY ?? null,
}

