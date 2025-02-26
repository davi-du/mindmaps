import { ChatMistralAI } from "@langchain/mistralai";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "langchain/vectorstores/memory"; 
import dotenv from 'dotenv'; // Libreria per caricare le variabili d'ambiente

dotenv.config();    // Carica le variabili d'ambiente dal file .env

//modello di chat 
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0    //anche qui risposte più deterministiche come per openAI?
});

//modello di embedding
const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed"
});

//Inizializza il Vector Store in memoria
const vectorStore = new MemoryVectorStore(embeddings);

export {
    llm,           // Modello di chat OpenAI
    embeddings,    // Modello di embedding OpenAI
    vectorStore,   // Vector Store per salvare i documenti
};
