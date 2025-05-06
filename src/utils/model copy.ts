import { debug, userProvidedAPIKey } from '../constants'

export const ModelAvailable = 'codestral-2405'  //globalBestModelAvailable

export type ModelForMagic = 'codestral-2405' | 'codestral-latest'

export const models = {
  smarter: ModelAvailable as ModelForMagic,
  faster: ModelAvailable as ModelForMagic,
}

const temperatures = {
  response: 0.7,
  parsing: 0.3,
}

export interface AIChatCompletionResponseStream {
  id: string
  object: string
  created: number
  model: string
  choices: {
    delta: {
      content?: string
      role?: string
    }
  }[]
  index: number
  finish_reason: string
}

export interface Prompt {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export const getCompletionOptions = (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature: number | undefined,
  token: number | undefined,
  stream = false,
) => {
  return {
    messages: prompts,
    ////
    model: model,
    temperature: temperature,
    max_tokens: token,
    n: 1,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    stream: stream,
  }
}

const getRequestOptions = (options: any) => {
  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization:
        'Bearer ' +
        String(
          debug
            ? process.env.REACT_APP_OPENAI_API_KEY
            : userProvidedAPIKey.current,
        ),
    },
    body: JSON.stringify(options),
  }
}

export const getAICompletion = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.response,
  token = 1024,
) => {
  console.log(`asking ${model}`, prompts)

  const options = getCompletionOptions(prompts, model, temperature, token)
  const requestOptions = getRequestOptions(options)

  const response = await fetch(
    //'https://api.openai.com/v1/chat/completions',
    'https://api.mistral.ai/v1/chat/completions',
    requestOptions,
  )

  const data = await response.json()

  return data
}

export const streamAICompletion = async (
  prompts: Prompt[],
  model: ModelForMagic,
  streamFunction: (data: any, freshStream: boolean) => void,
  freshStream: boolean,
  temperature = temperatures.response,
  token = 2048,
) => {
  console.log(`streaming ${model}`, prompts)

  const options = getCompletionOptions(prompts, model, temperature, token, true)
  const requestOptions = getRequestOptions(options)

  const response = await fetch(
    //'https://api.openai.com/v1/chat/completions',
    'https://api.mistral.ai/v1/chat/completions',
    requestOptions,
  )

  const reader = response.body?.getReader()
  if (!reader) return

  const dataParticle = {
    current: '',
  }
  while (true) {
    const { done, value } = await reader.read()

    if (done) break

    const data = new TextDecoder('utf-8').decode(value)
    // response format
    // data: { ... }

    const dataObjects = data
      .split('\n')
      .map(d => d.trim())
      .filter(d => d.length > 0)
    dataObjects.forEach(d => {
      d = d.replace(/^data: /, '').trim()
      if (d === '[DONE]') return

      let parsingSuccessful = false
      try {
        const dataObject = JSON.parse(d) as AIChatCompletionResponseStream

        if (
          dataObject.choices?.length > 0 &&
          dataObject.choices[0].delta.content
        )
          streamFunction(dataObject, freshStream)

        parsingSuccessful = true
        dataParticle.current = ''
      } catch (error) {
        parsingSuccessful = false
        dataParticle.current += d
        // console.error('stream error', error, data)
      }

      if (!parsingSuccessful) {
        // try parsing dataParticle
        try {
          const dataObject = JSON.parse(
            dataParticle.current,
          ) as AIChatCompletionResponseStream

          if (
            dataObject.choices?.length > 0 &&
            dataObject.choices[0].delta.content
          )
            streamFunction(dataObject, freshStream)

          dataParticle.current = ''
        } catch (error) { }
      }
    })
  }

  return
}

/* -------------------------------------------------------------------------- */
/*
//mod mia
export const parseAIResponseToObjects = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.parsing,
  token = 2048,
) => {
  console.log(`parsing ${model}`, prompts)

  const options = getCompletionOptions(
    prompts,
    model,
    temperature,
    token,
    false,
  )
  const requestOptions = getRequestOptions(options)

  try {
    const response = await fetch(
      //'https://api.openai.com/v1/chat/completions',
      'https://api.mistral.ai/v1/chat/completions',
      requestOptions,
    )

    const data = await response.json()

    return data
  } catch (error) {
    return {
      error: error,
    }
  }
}
*/
/*
export const parseAIResponseToObjects = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.parsing,
  token = 2048,
) => {
  console.log(`Parsing ${model}`, prompts)

  const options = getCompletionOptions(prompts, model, temperature, token, false)
  const requestOptions = getRequestOptions(options)

  try {
    const response = await fetch(
      'https://api.mistral.ai/v1/chat/completions',
      requestOptions,
    )

    if (!response.ok) {
      console.error('API Request failed:', response.status, await response.text())
      return { error: `API request failed with status ${response.status}` }
    }

    const data = await response.json()

    if (!data || !data.choices || data.choices.length === 0) {
      console.error('Invalid API response:', data)
      return { error: 'Invalid API response structure', choices: [] }
    }

    return data
  } catch (error) {
    console.error('Error fetching AI response:', error)
    return { error: (error as Error).message || 'Unknown error', choices: [] }
  }
}
*/

export const parseAIResponseToObjects = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.parsing,
  token = 2048,
  retries = 3, // Numero massimo di tentativi
  delay = 2000, // Attesa iniziale (2 sec)
) => {
  //console.log(`Parsing ${model}`, prompts)
  console.log(`Parsing ${model}`)
  console.log("Prompts inviati:", JSON.stringify(prompts, null, 2))

  const options = getCompletionOptions(prompts, model, temperature, token, false)
  const requestOptions = getRequestOptions(options)

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetch(
        'https://api.mistral.ai/v1/chat/completions',
        requestOptions,
      )

      if (response.status === 429) { // Rate limit
        console.warn(`Rate limit exceeded. Retrying in ${delay}ms...`)
        await new Promise((resolve) => setTimeout(resolve, delay))
        delay *= 2 // Raddoppia il tempo di attesa ad ogni retry
        continue
      }

      if (!response.ok) {
        console.error('API Request failed:', response.status, await response.text())
        return { error: `API request failed with status ${response.status}`, choices: [] }
      }

      const data = await response.json()

      if (!data || !data.choices || data.choices.length === 0) {
        console.error('Invalid API response:', data)
        return { error: 'Invalid API response structure', choices: [] }
      }

      return data
    } catch (error) {
      console.error('Error fetching AI response:', error)
      if (attempt === retries - 1) return { error: (error as Error).message || 'Unknown error', choices: [] }
    }
  }

  return { error: 'Failed to get response from API', choices: [] }
}


/* -------------------------------------------------------------------------- */

/*
export const getTextFromModelResponse = (response: any): string => {
  if (response.error) return ''
  return response.choices![0]!.message.content ?? ''
}
*/

export const getTextFromModelResponse = (response: any): string => {
  if (!response || !response.choices || response.choices.length === 0) {
    console.error("Invalid API response", response);
    return '';
  }
  return response.choices[0]?.message?.content ?? '';
};

export const getTextFromStreamResponse = (
  response: AIChatCompletionResponseStream,
): string => {
  return response.choices[0].delta.content ?? ''
}

