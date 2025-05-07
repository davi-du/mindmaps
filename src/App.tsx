import React, { 
  createContext, 
  useEffect, 
  useState 
} from 'react'

import { ChatContext } from './components/Contexts'
import { Interchange } from './components/Interchange'
import { newQuestionAndAnswer } from './utils/chatUtils'

import {
  EdgeInformation,
  NodeInformation,
  RelationshipSaliency,
} from './utils/responseProcessing'

import { Prompt } from './utils/model'
import { ListDisplayFormat } from './components/Answer'

import logomio from './media/logo_mio.png'

/* ------------------------------------------------------------------- */

export interface OriginRange {
  start: number
  end: number
  answerObjectId: string
  nodeIds: string[]
}

export interface NodeEntityIndividual extends NodeInformation {
  originRange: OriginRange
  originText: string
}

export interface NodeEntity {
  id: string
  displayNodeLabel: string
  pseudo: boolean
  individuals: NodeEntityIndividual[]
}

export interface EdgeEntity extends EdgeInformation {
  originRange: OriginRange
  originText: string
}

export interface AnswerSlideObject {
  content: string
}

export type AnswerObjectEntitiesTarget = 'originText' | 'summary'

export interface SentenceInAnswer {
  originalText: string
  offset: number
  length: number
}

export interface AnswerObject {
  id: string
  originText: {
    content: string
    nodeEntities: NodeEntity[]
    edgeEntities: EdgeEntity[]
  }
  summary: {
    content: string
    nodeEntities: NodeEntity[]
    edgeEntities: EdgeEntity[]
  }
  slide: AnswerSlideObject
  answerObjectSynced: {
    listDisplay: ListDisplayFormat
    saliencyFilter: RelationshipSaliency
    collapsedNodes: string[]
    sentencesBeingCorrected: SentenceInAnswer[]
  }
  complete: boolean
}

interface ModelStatus {
  modelAnswering: boolean
  modelParsing: boolean
  modelAnsweringComplete: boolean
  modelParsingComplete: boolean
  modelError: boolean
  modelInitialPrompts: Prompt[]
}

export interface QuestionAndAnswerSynced {
  answerObjectIdsHighlighted: string[]
  answerObjectIdsHighlightedTemp: string[]
  answerObjectIdsHidden: string[]
  highlightedCoReferenceOriginRanges: OriginRange[]
  highlightedNodeIdsProcessing: string[]
  saliencyFilter: RelationshipSaliency
}

export interface QuestionAndAnswer {
  id: string
  question: string
  answer: string
  answerObjects: AnswerObject[]
  modelStatus: ModelStatus
  synced: QuestionAndAnswerSynced
}

export interface PartialQuestionAndAnswer {
  id?: string
  question?: string
  answer?: string
  answerObjects?: AnswerObject[]
  modelStatus?: Partial<ModelStatus>
  synced?: Partial<QuestionAndAnswerSynced>
}

export interface DebugModeContextType {
  debugMode: boolean
  setDebugMode: (debugMode: boolean) => void
}
export const DebugModeContext = createContext<DebugModeContextType>({} as DebugModeContextType)

export const ChatApp = () => {
  const [questionsAndAnswers, setQuestionsAndAnswers] = useState<QuestionAndAnswer[]>([])
  const [debugMode, setDebugMode] = useState<boolean>(false)

  useEffect(() => {
    if (questionsAndAnswers.length === 0) {
      setQuestionsAndAnswers([newQuestionAndAnswer()])
    } else if (questionsAndAnswers[questionsAndAnswers.length - 1].answer.length > 0) {
      setQuestionsAndAnswers(prev => [...prev, newQuestionAndAnswer()])
    }
  }, [questionsAndAnswers])

  useEffect(() => {
    requestAnimationFrame(() => {
      void document.body.offsetHeight
    })
  }, [questionsAndAnswers])

  return (
    <ChatContext.Provider
      value={{
        questionsAndAnswersCount: questionsAndAnswers.length,
        setQuestionsAndAnswers,
      }}
    >
      <DebugModeContext.Provider value={{ debugMode, setDebugMode }}>
        <div className="chat-app">
          <div className="interchange-item graphologue-logo">
            <img src={logomio} alt="Mindmaps" />
          </div>

          {
            questionsAndAnswers.map((qa, index) => {
              const hasGraphData = qa.answerObjects.some(
                a => a.originText.nodeEntities.length > 0 || a.originText.edgeEntities.length > 0
              )

              const isLast = index === questionsAndAnswers.length - 1
              const shouldRender = isLast || qa.answer.length > 0 || hasGraphData

              return shouldRender ? (
                <Interchange key={`interchange-${qa.id}`} data={qa} />
              ) : null
            })

          }
        </div>
      </DebugModeContext.Provider>
    </ChatContext.Provider>
  )
} 
