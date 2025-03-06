import { indexDocuments } from "./indexing.ts";
import { llm, embeddings, vectorStore } from "../app.ts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";


/* Indexing */
const allSplits = await indexDocuments("./pdf_files", "https://en.wikipedia.org/wiki/Interpreter_(computing)");


/* Template del prompt per generare risposte basate sul contesto */

const template = `Use the following pieces of context to answer the question at the end.  
If you don't know the answer, just say that you don't know, don't try to make up an answer.  
Use three sentences maximum and keep the answer as concise as possible.  

For each relevant term, append a token in the format $N1 directly next to the term.  
If a term belongs to a broader category, assign it a subtoken in the format $N1C1 (for categories) or $N1N1 (for specific instances).  
For example, 'car $N2' 'SUV $N2C1', and 'van $N2C2'.  
Similarly, specific items inherit their category, such as 'Ferrari $N2N1' and 'Ford $N2N2'.  

Always say "Grazie per averlo chiesto!" at the end of the answer.  

{context}  

Question: {question}  
Helpful Answer:`;

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["user", template],
]);


/* STATE GRAPH: Definizione del grafo per retrieval e generazione */
const StateAnnotation = Annotation.Root({
  question: Annotation,
  context: Annotation,
  answer: Annotation,
});
const retrieve = async (state) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

const generate = async (state) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};

const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();


/* DOCUMENT CLASSIFICATION: Classifico i documenti in sezioni */
const totalDocuments = allSplits.length;
const third = Math.floor(totalDocuments / 3);
allSplits.forEach((document, i) => {
  if (i < third) {
    document.metadata["section"] = "beginning";
  } else if (i < 2 * third) {
    document.metadata["section"] = "middle";
  } else {
    document.metadata["section"] = "end";
  }
});


/* VECTOR STORE QA: Creo un nuovo Vector Store per QA */
const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits);
const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});
const structuredLlm = llm.withStructuredOutput(searchSchema);


/* STATE GRAPH QA: Definizione del grafo con analisi della query */
const StateAnnotationQA = Annotation.Root({
  question: Annotation<string>,
  search: Annotation<z.infer<typeof searchSchema>>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const analyzeQuery = async (state) => {
  const result = await structuredLlm.invoke(state.question);
  return { search: result };
};

const retrieveQA = async (state) => {
  const filter = (doc) => doc.metadata.section === state.search.section;
  const retrievedDocs = await vectorStore.similaritySearch(
    state.search.query,
    2,
    filter
  );
  return { context: retrievedDocs };
};

const generateQA = async (state) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};

const graphQA = new StateGraph(StateAnnotationQA)
  .addNode("analyzeQuery", analyzeQuery)
  .addNode("retrieveQA", retrieveQA)
  .addNode("generateQA", generateQA)
  .addEdge("__start__", "analyzeQuery")
  .addEdge("analyzeQuery", "retrieveQA")
  .addEdge("retrieveQA", "generateQA")
  .addEdge("generateQA", "__end__")
  .compile();


  //TEST
/* TEST RETRIEVAL E GENERAZIONE */
let inputs = { question: "What is an interpreter?" };
const result = await graph.invoke(inputs);
console.log(`\nQuestion: ${result["question"]}`);
console.log(`\nAnswer: ${result["answer"]}`);


/* TEST QA SYSTEM */
let inputsQA = {
  question: "During the software development cycle, programmers make frequent changes to the source code, how can an interpreter help?",
};
let outputQA;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA.question);
if (outputQA && outputQA.generateQA && outputQA.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}

let inputsQA1 = {
  question: "What is the name of the technique where a program in L0 executes programs written in L1 by interpreting each instruction without first generating a new program in L0?",
  //question: "Tell me how does it work and what's the paragraph name where it is in the file.",
};
let outputQA1;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA1, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA1 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA1.question);
if (outputQA1 && outputQA1.generateQA && outputQA1.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA1.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}


//altra domanda
let inputsQA2 = {
  question: "Is MATLAB an interpreted language or a compiled language?",
};
let outputQA2;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA2, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA2 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA2.question);
if (outputQA2 && outputQA2.generateQA && outputQA2.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA2.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}

//altra domanda
let inputsQA3 = {
  question: "Can you summarize the key advantages of using it?",
};
let outputQA3;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA3, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA3 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA3.question);
if (outputQA3 && outputQA3.generateQA && outputQA3.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA3.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}