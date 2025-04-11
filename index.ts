// index.ts
import express from "express";
import type { Request, Response } from "express";
import cors from "cors";
import { buildRagContext } from "./rag.ts";
//import { doc } from "prettier";

const app = express();
const PORT = 3001;

console.log("\n\nStarting RAG server\n\n");

app.use(cors());
app.use(express.json());

app.post("/rag", async (req: Request, res: Response) => {
  const { question } = req.body;

  if (!question) return res.status(400).json({ error: "Missing question." });

  try {
    const question = "What is an interpreter?";
    const context = await buildRagContext(question);

    res.json({ context });
  } catch (err) {
    console.error("Error in /rag:", err);
    res.status(500).json({ error: "Failed to retrieve context." });
  }
});

app.listen(PORT, () => {
  console.log(`RAG server listening on http://localhost:${PORT}`);
});

export {}; // necessario per evitare errore TS1208 se usi require()
