import { ChatMistralAI } from "@langchain/mistralai";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { ChatGroq } from "@langchain/groq";
import { CohereEmbeddings } from "@langchain/cohere";
import { MemoryVectorStore } from "langchain/vectorstores/memory"; 
import dotenv from 'dotenv'; // Libreria per caricare le variabili d'ambiente

dotenv.config();    // Carica le variabili d'ambiente dal file .env

//test MISTRAL
/*
//modello di chat 
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0    //anche qui risposte più deterministiche come per openAI?
});

//modello di embedding
const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed"
});
*/

//groq+cohere
const llm = new ChatGroq({
  model: "mixtral-8x7b-32768",
  temperature: 0
});

const embeddings = new CohereEmbeddings({
  model: "embed-english-v3.0"
});

//Inizializza il Vector Store in memoria
const vectorStore = new MemoryVectorStore(embeddings);

/*
const response = await llm.call([
  { role: "user", content: "Dimmi una curiosità su Python." }
]);
console.log(response);
*/


export {
    llm,           // Modello di chat OpenAI
    embeddings,    // Modello di embedding OpenAI
    vectorStore,   // Vector Store per salvare i documenti
};
