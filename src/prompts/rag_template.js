import { ChatPromptTemplate } from "@langchain/core/prompts";

const template = `
Context information is below:
---------------
{context}
---------------

Given the context information and not prior knowledge, answer the following question:
Question: {question}

Answer: Let's approach this step by step:
`;

export const ragPrompt = ChatPromptTemplate.fromTemplate(template);