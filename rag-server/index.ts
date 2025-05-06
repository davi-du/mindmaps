// index.ts
import express from "express";
import type { Request, Response } from "express";
import cors from "cors";
import { buildRagContext } from "./rag.ts";
import { doc } from "prettier";

const app = express();
const PORT = 3001;

const FgYellow = "\x1b[33m";
const FgGreen = "\x1b[32m";

console.log(`\n ${FgYellow} *** Starting RAG server ***`);

app.use(cors());
app.use(express.json());

app.post("/rag", async (req: Request, res: Response) => {
  const { question } = req.body;

  if (!question) return res.status(400).json({ error: "Missing question." });

  try {
    const context = await buildRagContext(question);
    res.json({ context });
  } catch (err) {
    console.error("Error in /rag:", err);
    res.status(500).json({ error: "Failed to retrieve context." });
  }
});

app.listen(PORT, () => {
  console.log(`${FgGreen}RAG server listening on http://localhost:${PORT}\n\n`);
});

export {}; // necessario per evitare errore TS1208 se usi require()
