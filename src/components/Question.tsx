import React, {
  ChangeEvent,
  KeyboardEvent,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'

import SendRoundedIcon from '@mui/icons-material/SendRounded'
import HourglassTopRoundedIcon from '@mui/icons-material/HourglassTopRounded'
import ClearRoundedIcon from '@mui/icons-material/ClearRounded'
import ReplayRoundedIcon from '@mui/icons-material/ReplayRounded'

import { AnswerObject } from '../App'
import { ChatContext } from './Contexts'

import {
  getTextFromModelResponse,
  getTextFromStreamResponse,
  models,
  AIChatCompletionResponseStream,
  parseAIResponseToObjects,
  streamAICompletion,
} from '../utils/model'

import {
  predefinedPrompts,
  predefinedPromptsForParsing,
} from '../utils/prompts'

import {
  getAnswerObjectId,
  helpSetQuestionAndAnswer,
  newQuestionAndAnswer,
  trimLineBreaks,
} from '../utils/chatUtils'

import { InterchangeContext } from './Interchange'

import { SentenceParser, SentenceParsingJob } from '../utils/sentenceParser'

import {
  cleanSlideResponse,
  nodeIndividualsToNodeEntities,
  parseEdges,
  parseNodes,
  RelationshipSaliency,
  removeAnnotations,
  removeLastBracket,
} from '../utils/responseProcessing'

import { ListDisplayFormat } from './Answer'

import { debug } from '../constants'

import { makeFlowTransition } from '../utils/flowChangingTransition'
// import rag
import { getRagContext } from '../utils/ragClient'


///////////////////////////////////////////////////////////////////////////


export type FinishedAnswerObjectParsingTypes = 'summary' | 'slide'

export const Question = () => {
  const { questionsAndAnswersCount, setQuestionsAndAnswers } =
    useContext(ChatContext)
  const {
    questionAndAnswer: {
      id,
      question,
      answer,
      modelStatus: { modelAnswering, modelError, modelAnsweringComplete },
    },
    handleSelfCorrection,
  } = useContext(InterchangeContext)

  const { questionsAndAnswers } = useContext(ChatContext)
  const isSameQuestion = question === questionsAndAnswers[id]?.question
  const allQuestions = Object.values(questionsAndAnswers)
  const isLastQuestion = allQuestions[allQuestions.length - 1]?.id === id




  const [activated, setActivated] = useState(false) // show text box or not
  const questionItemRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!activated && (questionsAndAnswersCount < 2 || answer.length > 0)) {
      setActivated(true)
    }
  }, [activated, answer.length, questionsAndAnswersCount])

  /* -------------------------------------------------------------------------- */

  const canAsk = question.length > 0 && !modelAnswering

  /* -------------------------------------------------------------------------- */

  const autoGrow = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'fit-content'
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'
    }
  }, [])

  const handleChange = useCallback(
    (event: ChangeEvent) => {
      if (event.target instanceof HTMLTextAreaElement) {
        const newQuestion = event.target.value

        setQuestionsAndAnswers(prevQsAndAs =>
          helpSetQuestionAndAnswer(prevQsAndAs, id, {
            question: newQuestion,
          }),
        )

        autoGrow()
      }
    },
    [autoGrow, id, setQuestionsAndAnswers],
  )

  useEffect(() => {
    autoGrow()
  }, [autoGrow])

  /* -------------------------------------------------------------------------- */

  // ! smart part
  const answerStorage = useRef<{
    answer: string
    answerObjects: AnswerObject[]
  }>({
    answer: '', // raw uncleaned text
    answerObjects: [],
  })

  const handleResponseError = useCallback(
    (response: any) => {
      console.error(response.error)

      setQuestionsAndAnswers(prevQsAndAs =>
        helpSetQuestionAndAnswer(prevQsAndAs, id, {
          // answerObjects: [], // ?
          modelStatus: {
            modelError: true,
          },
        }),
      )
    },
    [id, setQuestionsAndAnswers],
  )

  const handleSentenceParsingResult = useCallback(
    (result: SentenceParsingJob) => {
      const { sourceAnswerObjectId } = result

      const sourceAnswerObject = answerStorage.current.answerObjects.find(
        answerObject => answerObject.id === sourceAnswerObjectId,
      )
      if (!sourceAnswerObject || sourceAnswerObject.complete)
        // do not touch complete answer objects
        return

      answerStorage.current.answerObjects =
        answerStorage.current.answerObjects.map((a: AnswerObject) => {
          if (a.id === sourceAnswerObjectId) {
            return {
              ...a,
              complete: true,
            }
          } else return a
        })

      setQuestionsAndAnswers(prevQsAndAs =>
        helpSetQuestionAndAnswer(prevQsAndAs, id, {
          answerObjects: answerStorage.current.answerObjects,
        }),
      )
    },
    [id, setQuestionsAndAnswers],
  )

  const sentenceParser = useRef<SentenceParser>(
    new SentenceParser(handleSentenceParsingResult, handleResponseError),
  )

  /* -------------------------------------------------------------------------- */
  // ! stream graph

  const _groundRest = useCallback(() => {
    setQuestionsAndAnswers(prevQsAndAs =>
      helpSetQuestionAndAnswer(
        prevQsAndAs,
        id,
        newQuestionAndAnswer({
          id,
          question,
          modelStatus: {
            modelAnswering: true,
            modelParsing: true, // parsing starts the same time as answering
          },
        }),
      ),
    )
    answerStorage.current.answer = ''
    answerStorage.current.answerObjects = []

    sentenceParser.current.reset() // ? still needed?

    textareaRef.current?.blur()

    // scroll to the question item (questionItemRef)
    setTimeout(() => {
      const answerWrapper = document.querySelector(
        `.answer-wrapper[data-id="${id}"]`,
      )
      if (answerWrapper)
        answerWrapper.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
          inline: 'nearest',
        })
      // if (questionItemRef.current) {
      //   // find element with className .answer-wrapper and data-id = id
      //   questionItemRef.current.scrollIntoView({
      //     behavior: 'smooth',
      //   })
      // }
    }, 1000)
  }, [id, question, setQuestionsAndAnswers])

  /* -------------------------------------------------------------------------- */

  const handleUpdateRelationshipEntities = useCallback(
    (content: string, answerObjectId: string) => {
      const answerObject = answerStorage.current.answerObjects.find(
        a => a.id === answerObjectId,
      )
      if (!answerObject) return

      const cleanedContent = removeLastBracket(content, true)
      const nodes = parseNodes(cleanedContent, answerObjectId)
      const edges = parseEdges(cleanedContent, answerObjectId)

      answerStorage.current.answerObjects =
        answerStorage.current.answerObjects.map(
          (a: AnswerObject): AnswerObject => {
            if (a.id === answerObjectId) {
              return {
                ...a,
                originText: {
                  ...a.originText,
                  nodeEntities: nodeIndividualsToNodeEntities(nodes),
                  edgeEntities: edges,
                },
              }
            } else return a
          },
        )

      // setQuestionsAndAnswers(prevQsAndAs =>
      //   helpSetQuestionAndAnswer(prevQsAndAs, id, {
      //     answerObjects: answerStorage.current.answerObjects,
      //   })
      // )
    },
    [],
  )

  /* -------------------------------------------------------------------------- */

  //LLM CALL
  // ! self correction
  const handleParsingCompleteAnswerObject = useCallback(
    async (answerObjectId: string) => {
      const answerObjectToCorrect = answerStorage.current.answerObjects.find(
        a => a.id === answerObjectId,
      )
      if (!answerObjectToCorrect) return

      // self correction
      const correctedOriginTextContent = await handleSelfCorrection(
        answerObjectToCorrect,
      )
      answerStorage.current.answer = answerStorage.current.answer.replace(
        answerObjectToCorrect.originText.content,
        correctedOriginTextContent,
      )
      /*
       // originale
       answerObjectToCorrect.originText.content = correctedOriginTextContent
       handleUpdateRelationshipEntities(
         correctedOriginTextContent,
         answerObjectId,
       )
         */

      // Riannotazione se nodi vuoti DA RIVEDERE
      answerObjectToCorrect.originText.content = correctedOriginTextContent
      const tempNodes = parseNodes(correctedOriginTextContent, answerObjectId)
      if (tempNodes.length === 0) {
        console.warn('[Riannotazione] Nessun nodo rilevato, forzo parsing...')
      }
      handleUpdateRelationshipEntities(correctedOriginTextContent, answerObjectId)

      // set corrected answer object
      setQuestionsAndAnswers(prevQsAndAs =>
        helpSetQuestionAndAnswer(prevQsAndAs, id, {
          answer: answerStorage.current.answer,
          answerObjects: answerStorage.current.answerObjects, // TODO account for answerObjectSynced changes
        }),
      )

      /* -------------------------------------------------------------------------- */
      // parse slides and summary
      const answerObject = answerStorage.current.answerObjects.find(
        a => a.id === answerObjectId,
      )
      if (!answerObject) return

      const parsingResults: {
        [key in FinishedAnswerObjectParsingTypes]: string
      } = {
        summary: '',
        slide: '',
      }

      let parsingError = false
      await Promise.all(
        (['summary', 'slide'] as FinishedAnswerObjectParsingTypes[]).map(
          async (parsingType: FinishedAnswerObjectParsingTypes) => {
            if (parsingError) return

            const parsingSummary = parsingType === 'summary'

            // ! request

            const parsingResult = await parseAIResponseToObjects(
              predefinedPromptsForParsing[parsingType](
                parsingSummary
                  ? answerObject.originText.content
                  : removeAnnotations(answerObject.originText.content),
              ),
              debug ? models.faster : models.smarter,
            )

            if (parsingResult.error) {
              console.error('Error in parsing result:', parsingResult.error)
              parsingError = true
              return
            }

            if (!parsingResult.choices || parsingResult.choices.length === 0) {
              console.error('No valid choices returned from API:', parsingResult)
              parsingError = true
              return
            }

            parsingResults[parsingType] = getTextFromModelResponse(parsingResult)

          },


        ),
      )

      if (!parsingError) {
        // ! complete answer object
        answerStorage.current.answerObjects =
          answerStorage.current.answerObjects.map((a: AnswerObject) => {
            if (a.id === answerObjectId) {
              return {
                ...a,
                summary: {
                  content: parsingResults.summary,
                  nodeEntities: nodeIndividualsToNodeEntities(
                    parseNodes(parsingResults.summary, answerObjectId),
                  ),
                  edgeEntities: parseEdges(
                    parsingResults.summary,
                    answerObjectId,
                  ),
                },
                slide: {
                  content: cleanSlideResponse(parsingResults.slide),
                },
                complete: true,
              }
            } else return a
          })

        setQuestionsAndAnswers(prevQsAndAs =>
          helpSetQuestionAndAnswer(prevQsAndAs, id, {
            answerObjects: answerStorage.current.answerObjects,
            // modelStatus: {
            //   modelAnswering: false,
            //   modelAnsweringComplete: true,
            //   modelParsing: false,
            //   modelParsingComplete: true,
            // },
          }),
        )
      }
    },
    [
      handleResponseError,
      handleSelfCorrection,
      handleUpdateRelationshipEntities,
      id,
      setQuestionsAndAnswers,
    ],
  )

  const handleStreamRawAnswer = useCallback(
    (data: AIChatCompletionResponseStream, freshStream = true) => {
      const deltaContent = trimLineBreaks(getTextFromStreamResponse(data))
      if (!deltaContent) return

      const aC = answerStorage.current

      // this is the first response streamed
      const isFirstAnswerObject = aC.answerObjects.length === 0
      const hasLineBreaker = deltaContent.includes('\n')

      let targetLastAnswerObjectId = aC.answerObjects.length
        ? aC.answerObjects[aC.answerObjects.length - 1].id
        : null

      // ! ground truth of the response
      aC.answer += deltaContent
      // sentenceParser.current.updateResponse(aC.answer)

      const _appendContentToLastAnswerObject = (content: string) => {
        const lastObject = aC.answerObjects[aC.answerObjects.length - 1]
        lastObject.originText.content += content
      }

      const preparedNewObject = {
        id: getAnswerObjectId(), // add id
        summary: {
          content: '',
          nodeEntities: [],
          edgeEntities: [],
        }, // add summary
        slide: {
          content: '',
        }, // pop empty slide
        answerObjectSynced: {
          listDisplay: 'original' as ListDisplayFormat,
          saliencyFilter: 'high' as RelationshipSaliency,
          collapsedNodes: [],
          sentencesBeingCorrected: [],
        },
        complete: false,
      }

      // break answer into parts
      if (isFirstAnswerObject) {
        // * new answer object
        aC.answerObjects.push({
          ...preparedNewObject,
          originText: {
            content: deltaContent,
            nodeEntities: [],
            edgeEntities: [],
          },
        })

        targetLastAnswerObjectId = preparedNewObject.id
        ////
      } else if (hasLineBreaker) {
        // add a new answer object
        const paragraphs = deltaContent
          .split('\n')
          .map(c => c.trim())
          .filter(c => c.length)

        let paragraphForNewAnswerObject = ''

        if (paragraphs.length === 2) {
          paragraphForNewAnswerObject = paragraphs[1]
          ////
          // if (!isFirstAnswerObject)
          _appendContentToLastAnswerObject(paragraphs[0])
        } else if (paragraphs.length === 1) {
          if (deltaContent.indexOf('\n') === 0)
            paragraphForNewAnswerObject = paragraphs[0]
          else {
            // if (!isFirstAnswerObject)
            _appendContentToLastAnswerObject(paragraphs[0])
          }
        } else {
          // do nothing now
        }

        // * new answer object
        aC.answerObjects.push({
          ...preparedNewObject,
          // originRange: {
          //   start: aC.answer.length - paragraphForNewAnswerObject.length,
          //   end: aC.answer.length,
          // }, // from text to ranges
          originText: {
            content: paragraphForNewAnswerObject,
            nodeEntities: [],
            edgeEntities: [],
          }, // add raw text
        })

        // ! finish a previous answer object
        // as the object is finished, we can start parsing it
        // adding summary, slide, relationships
        handleParsingCompleteAnswerObject(
          aC.answerObjects[aC.answerObjects.length - 2].id,
        )

        targetLastAnswerObjectId = preparedNewObject.id
        ////
      } else {
        // append to last answer object
        _appendContentToLastAnswerObject(deltaContent)
      }

      // ! parse relationships right now
      const lastParagraph = aC.answer.split('\n').slice(-1)[0]
      if (targetLastAnswerObjectId)
        handleUpdateRelationshipEntities(
          lastParagraph,
          targetLastAnswerObjectId,
        )

      // parse sentence into graph RIGHT NOW
      // if (deltaContent.includes('.')) {
      //   // get the last sentence from answerStorage.current.answer
      //   const dotAndBefore = deltaContent.split('.').slice(-2)[0] + '.'
      //   const lastSentencePartInPrevAnswerStorage = prevAnswerStorage
      //     .split('.')
      //     .slice(-2)[0]
      //   const lastSentence = lastSentencePartInPrevAnswerStorage + dotAndBefore

      //   // parse it
      //   if (prevLastAnswerObjectId) {
      //     sentenceParser.current.addJob(lastSentence, prevLastAnswerObjectId)
      //   }
      // }

      // * finally, update the state
      setQuestionsAndAnswers(prevQsAndAs =>
        helpSetQuestionAndAnswer(prevQsAndAs, id, {
          answer: aC.answer,
          answerObjects: aC.answerObjects, // TODO account for answerObjectSynced changes
        }),
      )

      // scroll
      // TODO
      /*
      const lastAnswerObject = aC.answerObjects[aC.answerObjects.length - 1]
      const answerObjectElement = document.querySelector(
        `.answer-text[data-id="${lastAnswerObject.id}"]`
      )
      if (answerObjectElement) {
        answerObjectElement.scrollIntoView({
          behavior: 'smooth',
          block: 'end',
        })
      }
      */
    },
    [
      handleParsingCompleteAnswerObject,
      handleUpdateRelationshipEntities,
      id,
      setQuestionsAndAnswers,
    ],
  )

  const handleReprocess = useCallback(async () => {
    _groundRest() // se serve a gestire lo stato UI

    const ragContext = await getRagContext(question)

    const prompts =
      ragContext === "I DON'T KNOW" || ragContext === "I DON'T KNOW."
        ? predefinedPrompts.initialAsk(question)
        : predefinedPrompts.initialAskRag(`${ragContext}\n\n${question}`)

    const response = await parseAIResponseToObjects(prompts, debug ? models.faster : models.smarter)
    const regeneratedText = getTextFromModelResponse(response)

    makeFlowTransition()

    const corrected = await handleSelfCorrection({
      id,
      originText: { content: regeneratedText, nodeEntities: [], edgeEntities: [] },
      summary: { content: '', nodeEntities: [], edgeEntities: [] },
      slide: { content: '' },
      answerObjectSynced: {
        listDisplay: 'original',
        saliencyFilter: 'high',
        collapsedNodes: [],
        sentencesBeingCorrected: [],
      },
      complete: false,
    })

    const nodes = parseNodes(corrected, id)
    const edges = parseEdges(corrected, id)

    const [summaryResp, slideResp] = await Promise.all([
      parseAIResponseToObjects(
        predefinedPromptsForParsing.summary(removeAnnotations(corrected)),
        debug ? models.faster : models.smarter
      ),
      parseAIResponseToObjects(
        predefinedPromptsForParsing.slide(removeAnnotations(corrected)),
        debug ? models.faster : models.smarter
      ),
    ])

    const summaryText = getTextFromModelResponse(summaryResp)
    const slideText = getTextFromModelResponse(slideResp)

    const newAnswerObject: AnswerObject = {
      id,
      originText: {
        content: corrected,
        nodeEntities: nodeIndividualsToNodeEntities(nodes),
        edgeEntities: edges,
      },
      summary: {
        content: summaryText,
        nodeEntities: nodeIndividualsToNodeEntities(parseNodes(summaryText, id)),
        edgeEntities: parseEdges(summaryText, id),
      },
      slide: {
        content: cleanSlideResponse(slideText),
      },
      answerObjectSynced: {
        listDisplay: 'original',
        saliencyFilter: 'high',
        collapsedNodes: [],
        sentencesBeingCorrected: [],
      },
      complete: true,
    }

    setQuestionsAndAnswers(prev =>
      helpSetQuestionAndAnswer(prev, id, {
        answer: corrected,
        answerObjects: [newAnswerObject],
        modelStatus: {
          modelAnswering: false,
          modelAnsweringComplete: true,
          modelParsing: false,
          modelParsingComplete: true,
          modelInitialPrompts: [...prompts.map(p => ({ ...p }))],
        },
      })
    )
  }, [question, id, setQuestionsAndAnswers])



  const handleAskStream = useCallback(async () => {
    let hasStarted = false

    _groundRest()

    // recupera contesto
    const ragContext = await getRagContext(question)

    /*
    se il recupero delle informazioni con la rag va a buon fine o meno cambia il prompt
      - se il contesto della rag è "I don't know." (significa che il recupero non è andato 
        a buon fine) il modello genererà la risposta
      - se la rag invece ha generato un contesto il modello si limiterà a tokenizzarlo,
        questo perché la rag dopo aver recuperato le info rilevanti chiama il modello per generare 
        direttamente la risposta per allegerire il carico di lavoro di graphologue
        (è stato aggiunto un prompt nell'apposito file per questo caso,
        la rag ne ha uno apposito per generare la risposta corretta)
    */

    let tmpPrompts = null;

    if (ragContext === "I DON'T KNOW" || ragContext === "I DON'T KNOW.") {
      console.log(`- RAG context not retrieved -`);
      const contextWithQuestion = question;
      tmpPrompts = predefinedPrompts.initialAsk(contextWithQuestion);
    } else {
      console.log(`- RAG context retrieved successfully -`);
      const contextWithQuestion = `${ragContext}\n\n${question}`;
      tmpPrompts = predefinedPrompts.initialAskRag(contextWithQuestion);
    }

    const initialPrompts = tmpPrompts;

    const answerObjectId = getAnswerObjectId()
    const preparedAnswerObject: AnswerObject = {
      id: answerObjectId,
      originText: {
        content: '',
        nodeEntities: [],
        edgeEntities: [],
      },
      summary: {
        content: '',
        nodeEntities: [],
        edgeEntities: [],
      },
      slide: {
        content: '',
      },
      answerObjectSynced: {
        listDisplay: 'original',
        saliencyFilter: 'high',
        collapsedNodes: [],
        sentencesBeingCorrected: [],
      },
      complete: false,
    }

    answerStorage.current.answer = ''
    answerStorage.current.answerObjects = [preparedAnswerObject]

    // risposta in un solo paragrafo
    await streamAICompletion(
      initialPrompts,
      debug ? models.faster : models.smarter,
      (data, fresh) => {
        const delta = trimLineBreaks(getTextFromStreamResponse(data))
        if (!delta) return

        // ✅ Primo token ricevuto
        if (!hasStarted) {
          hasStarted = true
          makeFlowTransition()

          setQuestionsAndAnswers(prev =>
            helpSetQuestionAndAnswer(prev, id, {
              answer: delta,
              answerObjects: [{
                ...preparedAnswerObject,
                originText: {
                  ...preparedAnswerObject.originText,
                  content: delta,
                },
              }],
            })
          )
          return
        }

        // ✅ Token successivi → aggiornamento continuo
        answerStorage.current.answer += delta
        answerStorage.current.answerObjects[0].originText.content += delta

        setQuestionsAndAnswers(prev => helpSetQuestionAndAnswer(prev, id, {
          answer: answerStorage.current.answer,
          answerObjects: answerStorage.current.answerObjects,
        }))
      }, true
    )

    await handleParsingCompleteAnswerObject(answerObjectId)

    setQuestionsAndAnswers(prev => helpSetQuestionAndAnswer(prev, id, {
      answer: answerStorage.current.answer,
      answerObjects: answerStorage.current.answerObjects,
      modelStatus: {
        modelAnswering: false,
        modelAnsweringComplete: true,
        modelParsing: false,
        modelParsingComplete: true,
        modelInitialPrompts: [...initialPrompts.map(p => ({ ...p }))],
      },
    }))
  }, [
    _groundRest,
    getRagContext,
    setQuestionsAndAnswers,
    helpSetQuestionAndAnswer,
    id,
    question,
    debug,
    handleParsingCompleteAnswerObject,
  ])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // if cmd + enter
      //if (e.key === 'Enter' && e.metaKey) {
      if (e.key === 'Enter') {
        if (canAsk) handleAskStream()
      }
    },
    [canAsk, handleAskStream],
  )

  const handleDeleteInterchange = useCallback(() => {
    setQuestionsAndAnswers(prevQsAndAs =>
      prevQsAndAs.filter(qAndA => qAndA.id !== id),
    )
  }, [id, setQuestionsAndAnswers])



  return (
    <div
      ref={questionItemRef}
      className="question-item interchange-component drop-up"
    >
      {activated ? (
        <>
          <textarea
            ref={textareaRef}
            className="question-textarea"
            value={question}
            placeholder="ask a question"
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={!isLastQuestion}
          />

          <button
            className="bar-button"
            onClick={isSameQuestion ? handleReprocess : handleAskStream}
            disabled={
              modelAnswering ||
              (!question.trim()) ||  // nessuna domanda scritta
              (!isSameQuestion && !isLastQuestion) // blocca Ask se non sei sull'ultimo
            }
            title={isSameQuestion ? 'Reprocess' : 'Ask'}
          >
            {modelAnswering ? (
              <HourglassTopRoundedIcon className="loading-icon" />
            ) : isSameQuestion ? (
              <ReplayRoundedIcon />
            ) : (
              <SendRoundedIcon />
            )}
          </button>





          <button
            disabled={questionsAndAnswersCount < 2}
            className="bar-button"
            onClick={handleDeleteInterchange}
          >
            <ClearRoundedIcon />
          </button>
        </>
      ) : (
        <span
          className="new-question-hint"
          onClick={() => {
            setActivated(true)
          }}
        >
          add question
        </span>
      )}
      {modelError && (
        <div className="error-message">Got an error, please try again.</div>
      )}
    </div>
  )
}
