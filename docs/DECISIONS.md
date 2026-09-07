# Design & Governance Decisions

> **Last updated:** August 16, 2026 · **Framework version:** 0.2

This is a living document that helps to explain the rationale behind desired versus expected functionality in this AI framework project. I ultimately pursued this project as an overall learning tool, and going over these questions allows me to reflect upon the learnings obtained through each revision. These ten questions are drawn from the project's research brief ("Practical Research Questions to Answer Next").

## Contents

1. [Daily-life use cases](#1-what-are-three-daily-life-use-cases)
2. [Autonomy classification](#2-which-use-cases-are-read-only-drafting-only-supervised-action-or-autonomous)
3. [Sensitive data](#3-what-data-is-too-sensitive-to-leave-the-local-network)
4. [Systems never accessed](#4-which-systems-should-the-assistant-never-access)
5. [Tools needing approval](#5-what-tools-need-approval-before-execution)
6. [Logging](#6-what-should-be-logged-for-every-action)
7. [Reviewing & deleting memory](#7-how-will-stored-memories-be-reviewed-and-deleted)
8. [Failure recovery](#8-how-will-the-assistant-recover-from-a-failed-or-partial-workflow)
9. [Evaluating outputs](#9-how-will-model-outputs-be-evaluated-for-correctness-and-safety)
10. [Replaceable components](#10-which-component-should-be-replaceable-first-the-model-the-ui-the-vector-store-the-tool-layer-or-the-orchestration)

---

## 1. What are three daily-life use cases?

The current rendition of the AI Framework is a Self-Hosted interface for a single user looking for local AI assistance. My goal was for it to be useful for daily work while keeping the reach of its autonomy deliberately bounded and something that is controllable. I would say these are its current daily use cases.

1. **Cited Q&A Over Documents.** The framework is designed to ingest personal notes and files into a local knowledge base, which then a user can ask questions about them. The assistant then retrieves the most relevant messages, passages, and answers, grounded in strictly those sources, citing them by number, and saying plainly when the answer isn't present rather than inventing one. At minimum, I wanted this assistant to instill confidence that the answers provided are relevant and traceable to real source material.
2. **Durable Memory - Preference and Fact preservation.** I also wanted the assistant to ingest facts and/or preferences that the user tells it to remember. The assistant should then be able to remember that information across sessions, and even pull up the information when requested. As of today, these memories are stored in a human-readable format that can be listed, updated, or deleted at any time. The initial goal was to simply have the function available, and it currently doesn't consider things like additional privacy and security within each memory saved. The memory is inspectable and under the user's control.
3. **Local Drafting.** The assistant is able to help draft documents, emails, notes, summaries, etc. The assistant will write out a draft in accordance to the user's request as best as possible. But it is currently not designed to ever send or share anything on the user's behalf. The user must still make the final decision on whether or not the draft that has been created is sufficient for their needs or if they wish to manually edit accordingly.

These are three basic core capabilities of an assistant, all related under the concept of retrieval, memory, and generation (RAG), all while staying entirely local within a read-only or drafting safety envelope. The AI framework is also currently designed to be entirely local, using local LLMs by self-hosted models or through models obtained via Ollama. At present, the AI framework is not designed to reach out externally via the Internet.

### Beyond the MVP - Daily-life Use Cases

The initial goal in producing this framework was to get the base functions down. We have the model runtime, retrieval, the agent loop, the tools, and the memory in its place. I currently consider this the foundation for potential higher autonomy capabilities that would deliberately sit behind approval gates and an audit log. Real-world actions such as sending drafted emails, adding calendar events, or even retrieval from online sources for analysis and research... these are intentionally kept out of scope for this current version of the MVP.

---

## 2. Which use cases are read-only, drafting-only, supervised-action, or autonomous?

If we operate on this general behavioral pattern of autonomy:

| Level | Capability           | Relative example                                                       |
| :---: | -------------------- | ---------------------------------------------------------------------- |
|   0   | Chat only            | Explain a concept.                                                     |
|   1   | Read-only assistant  | Summarize today's calendar or a folder of notes.                       |
|   2   | Drafting assistant   | Draft an email, task list, or plan without sending it.                 |
|   3   | Supervised action    | Create an event only after explicit approval.                          |
|   4   | Bounded automation   | Every morning, prepare a daily briefing from approved sources.         |
|   5   | Proactive suggestion | Notice a schedule conflict and propose a fix.                          |
|   6   | Autonomous operator  | Take approved categories of low-risk actions without asking each time. |

I would say that intentionally the core of this AI framework is currently designed for Levels 0 to 2. That means for the future MVP state, I would be traversing into Level 3 or higher, where actions taken on behalf of the user must be taken only after explicit approval, supervised and critically analyzed. Keeping the initial and core functionality within level two and lower is a deliberate safety choice for a learning project such as this. Safety in this case is inherited by the limited capability, which only engages personal documentation, and drafts that have to be manually reviewed and sent.

We can see in the graph below that every capability in the current system falls into the lowest three rungs of the autonomy ladder (Levels 0 to 2).

| Use case / command                          | Autonomy level                   | Why it sits there                                                                                                                                                                                      |
| ------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `chat` (plain conversational REPL)          | Level 0: Chat only               | Answers from the model alone; no tools, no retrieval, no side effects.                                                                                                                                 |
| `ask` (grounded, cited Q&A)                 | Level 1: Read-only               | Retrieves from the local knowledge base and answers from those sources; only reads data.                                                                                                               |
| `agent` (model decides which tools to call) | Level 1: Read-only               | The agent may choose to search knowledge or recall memory, but every tool it can reach is read-only or writes only inside `data/`. Deciding to use a read-only tool does not raise the autonomy level. |
| `remember` / `recall` + `memory` CLI        | Level 1: Read-only (local state) | Reads and writes the assistant's own memory store; it never acts on the outside world.                                                                                                                 |
| `draft_document`                            | Level 2: Drafting                | Produces an artifact (an email, note, etc.) as a local file for me to review. It never sends or shares anything.                                                                                       |

---

## 3. What data is too sensitive to leave the local network?

In this current rendition of the AI framework, it is designed so that every category of user data is too sensitive to leave the machine. At its base, there is no cloud service, there is no external API, and there is no network egress in normal operation. The inference itself runs **locally** via self-hosted model (via Ollama). Nothing leaves the local instance.

The current design holds data that is open to the user to view. While it gives the user the ability to govern and control our data easily, additional considerations and protections would have to be made before I ever considered exposing the framework to network capabilities. At the very least, this data would have to be protected.

| Data                                      | Where it lives                                      | Why it's sensitive                                                                                                     |
| ----------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Ingested documents (personal notes/files) | `knowledge_base/`, embedded into `data/chroma/`     | The raw personal material the whole assistant reasons over.                                                            |
| Memories (preferences, personal facts)    | `data/memory.json` + the `memory` Chroma collection | Durable personal details I've explicitly asked it to retain.                                                           |
| Drafts (emails, notes)                    | `data/drafts/`                                      | Content intended for people/contexts, some of it private.                                                              |
| Traces (the audit log)                    | `data/traces/*.jsonl`                               | Records what I asked and when, sensitive even though it's "just logs," because the questions themselves reveal intent. |

### Mechanism

There are two layers to consider in this current implementation:

1. **Runtime.** The LLMs and embeddings run locally through Ollama. The vector store is an embedded, on-disk Chroma client with no server, and no tool performs any network I/O. As such, there is no user data off of the machine.
2. **Repository.** I'm using `.gitignore` which excludes the knowledge base and data directories that I utilized for testing. Though I haven't stored any sensitive content within those directories, the `.gitignore` is there to additionally ensure none of these sensitive categories can be committed to a public repository by accident.

I elected that this is sufficient for now since this is an initial release of my personal project. Future renditions of this project will naturally consider greater protections of personal data, ensuring any boundary crossings from local to external network are met by explicit, loud, approved decisions by the end user.

---

## 4. Which systems should the assistant never access?

At present, the answer to this is similar to question 3. In future renditions of this project, this section will likely shift into its own behavioral pattern. As stated previously, future renditions of this project will naturally consider greater protections of personal data as the consideration for the AI framework to access information via an external network is considered in parallel.

Today, the assistant doesn't access any external systems. Its world presently is the local model runtime and any files found in the data directory.

| System / resource                                  | Status                                                                                                      |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| The network / internet                             | Never - no outbound requests except to the local Ollama server (`localhost`).                               |
| Email (send or read)                               | Never: it can draft an email to a file, but has no mail access.                                             |
| Calendar / scheduling                              | Never: no calendar integration exists.                                                                      |
| The filesystem outside `data/`                     | Never: the only writing tool (`draft_document`) is confined to `data/drafts/`; nothing else writes to disk. |
| Shell / arbitrary command execution                | Never: there is no tool that runs OS commands or executes code.                                             |
| Credentials, secrets, financial or payment systems | Never: the assistant holds and touches none of these.                                                       |
| Other users' / household members' data             | N/A: single-user by design.                                                                                 |

### Mechanism

We can think of the boundary as a **closed allowlist** instead of a _blocklist_. The assistant can only do what a registered tool lets it perform. Those are four tools:

1. Search knowledge (`search_knowledge`)
2. Remember (`remember`)
3. Recall (`recall`)
4. Draft Document (`draft_document`)

Out of these, `draft_document` sanitizes its filename to a bare name (`Path(filename).name`), so even the one writing tool cannot escape data/drafts/ via path traversal. I made sure to test this in `tests/test_safety.py`.

It's important to realize that some of these walls are purposefully permanent in terms of future autonomy. For example, we wouldn't want the assistant to ever run arbitrary shell/command execution, direct access to credentials/secrets, or perform actions on the user's behalf without an audit trail. The concept of Level 3 - Supervised Actions ensures that a future state of this project considers events and actions done on the user's behalf only by explicit approval.

---

## 5. What tools need approval before execution?

As of today, **none**. None of the tools require a runtime approval step since none perform an action that would warrant one. Each toolset is either read-only or writes a file only to a local `data/` directory.

The closest action that could possibly be considered gate-worthy is the `draft_document` tool. However, the reason why this is not the case is because the output from this tool is a local and reversible file created that is not transmitted elsewhere. The user has the power to delete the file as well. By comparison, the natural place an approval gate would first attach to is where drafting a document graduates into sending a document on the user's behalf.

This speaks well to the future state of this AI framework. As soon as the assistant gets access to tools that require supervised action, those tools must be gated. The intention would be something like the following:

1. The agent proposes an action.
2. The system pauses and surfaces the exact action in its arguments.
3. The tool executes only after explicit user approval, with both the request and the decision written to the trace logs.

Tools that would require gating include:

1. Sending a drafted email or message.
2. creating or modifying a calendar event.
3. writing to files to other directories external to the `data/` directory.
4. Anything that touches upon external system or network.

### Mechanism

There are no approval gates implemented at this time since none of the tools created require it in this version. Tools have been verified through `tests/test_safety.py`.

---

## 6. What should be logged for every action?

From the beginning, **observability** has been a vital and core component into creating an AI framework of my choosing. From my perspective, nothing that the assistant does should be invisible. Every model interaction is recorded to an append-only, per-day JSONL trace (`data/traces/<date>.jsonl`). This ensures that any answer can be reconstructed after the fact and helps to provide an audit trail of why the assistant said or enacted an action in the way that it did. If I was going to build an assistant that acts on personal data, utmost trust depends on being able to see what the assistant performed. This is why observability is paramount in this project by principle.

Every `chat` / `ask` / `agent` command appends one event to the logs with the following information:

| Field                       | Purpose                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `timestamp` (UTC, ISO-8601) | when it happened                                              |
| `command`                   | which mode ran (chat / ask / agent)                           |
| `model`                     | which model produced the answer (model-choice accountability) |
| `question`                  | the exact user input                                          |
| `tools_available`           | which tools the agent could reach on that run                 |
| `retrieved` (for `ask`)     | the exact source chunks used, for citation auditing           |
| `answer_chars`, `latency_s` | response size and cost-in-time                                |

### Mechanism

Currently, for a base release, I elected that command level logging was enough for a level two system. Since the current scope of the assistant is that it takes no outward action, I've deferred recording each individual tool call and any failures in a fixed shape pattern like your standard typical programmer application. I am using `Tracer`, but lightly. `Tracer.record` simply takes a plain dict, provides a timestamp onto it, and writes the log.

As this project climbs the autonomy ladder, the logging should grow from a command level record into an action level audit trail. In my daily work life, I often deal with investigating logging, and so I'm generally familiar on what to look for. But otherwise this would be a new category for me to create something like this. At a high level, I can imagine logging every tool call that the agent makes. Perhaps the entire run of the agentic loop can be replayed step by step rather than inferring what may have occurred from the final answer. If I were to pursue this, I would want to always be able to answer clearly what the assistant did on the user's behalf and whether or not it was approved by the user. It is an interesting balance because at some point there would be additional considerations that need to be made for overall retention and redaction of log data.

---

## 7. How will stored memories be reviewed and deleted?

The current method is handled by user **decision**. I had designed memory from the ground up to be inspectable, editable, and forgettable, so that the user stays fully in control of what the assistant retains. The design as of today is meant for adding, reviewing, and deleting. There isn't an option for in-place editing for no other reason that I didn't get around to it.

### Mechanism

1. **Review (Inspection).** Data lives in `data/memory.json`. A user can open the file on any editor and see exactly what the assistant remembers. The entries include an ID, the text, and the creation timestamp. Running `memory list` from the CLI prints the same data.
2. **Delete (Removal).** The assistant forgets memory if the user executes `memory delete <id>`. The associated JSON record and its vector in the Chroma index (yes, there are two stores) get removed at the same time since they are keyed by the same shared ID. Deleting from both stores is essential because we wouldn't want a case where only deleting the JSON allowed a leftover vector to surface a forgotten fact in a recall.

As discussed, I don't have a function for an in-place memory update, but it doesn't seem out of scope to make a `memory edit` function. A future state here may include things like organizing memories by category or source, or an optional expiry for time-bound facts. A `memory recall` might be useful here too, so that the user can review what is currently being remembered for them and ensure that the memory that is there stays curated to what they currently need instead of silently accumulating forever.

---

## 8. How will the assistant recover from a failed or partial workflow?

This is also handled by a user **decision**. This is due to the AI framework's current simplicity in that the assistant runs no multi-step workflows that mutate external systems, so there is no rollback to consider. So a failure mid-run doesn't leave an external system in a partially finished state, such as a half sent email or partially created event. The worst case of any failure in this current implementation is an incomplete answer or an unwritten local file, which are both arguably harmless and reversible. Current error handling ensures that any failures are loud to the user. That being said, errors are not handled presently in a sustainable way. While recovery today is graceful for bad inputs, a future state requires much better error handling, especially when creating an infrastructure like a framework.

With that said, there are some guardrails that I have applied within this Level 2 infrastructure.

### Mechanism

1. **Runaway loops.** The agent loop is hard-capped via a setting I created called `config.max_tool_rounds`. If the model keeps requesting tools without settling on an answer, the loop stops after the cap and returns a clear fallback message instead of spinning forever.
2. **Bad tool calls.** Inside of the agent loop, a malformed tool request is caught (ex. unknown tool name, incorrect arguments) and turned into an error observation that is fed back to the model, so that a single bad call doesn't end up crashing the entire run. The idea is that the model can adjust during the next round.
3. **Invalid states.** There are fail fast guards which stop bad states at the source with clear messages rather than letting them corrupt data. For example: `add`/`recall` on a memory store built without a model defined raises an error immediately. Another example: An embedding-model mismatch on the vector store raises an error with a "re-index required" instruction instead of silently returning garbage.
4. **Low-risk local writes.** Memory and draft writes are simple and contained. A crash mid-write can at worst fail to save a draft or a memory.

If this project continues into the supervised action level of autonomy (Level 3) it will likely be that each tool needs its own recovery plan on how to detect a partial failure, how to undo or compensate, and how to record the attempt. It is also very likely that error handling will have to be designed alongside the approval gate and the per-action logging behavior. This is the only way that I can see that supervised actions within the AI framework are considered safe.

---

## 9. How will model outputs be evaluated for correctness and safety?

In terms of correctness, I designed it such that answers provided by the assistant can be checked manually. In terms of document Q&A, The assistant is instructed to answer only from the retrieved sources. From there, it is to cite those sources by number and to plainly say when the sources do not contain the answer. This ensures it does not invent one instead. In terms of safety, the current model output is considered low risk because it cannot trigger an action. The worst a bad answer can do is be wrong in what it provides the end user. Testing ensured that the tools are encapsulated by an allowlist, and the one write tool is not allowed to escape its single-defined destination.

### Mechanism

**Correctness** is verified through several layers:

1. **Grounded and Cited Answers.** The framework is designed to pull sources from a sources directory. From there, the assistant is able to pull knowledge from those sources and return an answer, citing the specific source chunks.
2. **Anti-hallucination guardrail.** Should the source not contain information about an out-of-corpus question (ex. looking for "airspeed of an unladen swallow" in the book Frankenstein), the assistant responds with "I don't know" or "I cannot find this" instead of producing a fabrication.
3. **Across session recall with citation.** Data that the user decides to save to memory is persistent across multiple sessions until the user decides to delete it. This was tested by providing one session a sentence to remember, and then having another session recall what that sentence was from memory.

This opens up an interesting retrospective in that correctness is currently checked manually at milestones. It's not as if there's some sort of automated regression set of questions that score the model's answers over time. This would be something that does need to be durable over time. Perhaps a "golden" set of Q&A questions that provide a handful of question/expected source pairs would be useful here to help detect any possible regression. Other ideas to ensure utmost quality could be automated grounding checks, and safety/red-team prompts that probe whether the model can be talked into misusing a tool.

---

## 10. Which component should be replaceable first? The model, the UI, the Vector Store, the tool layer, or the orchestration?

At first this seems like an odd question because I initially think of the AI framework holistically. In reality, this is more designed as a "modular monolith" in that this is a single deployable application that is internally decoupled into independent modules behind clean interfaces.

With that said, This AI framework was designed from the start with **replaceability** in mind. I knew from the beginning that technology would shift and improve over time. As such, my ranking in what needs to be replaced is as follows:

1. The Model
2. The Vector Store
3. The Application
4. The Tool Layer
5. The Orchestrator (the agent loop)

The model being first is what I would consider to be a natural occurrence. New and better local models appear constantly, so hard coding one would have been a very costly mistake. Ensuring that the AI framework is model agnostic was paramount to this project. While it does open up the ability to hot swap which model the user wishes to use. The ultimate goal was to ensure that the user was not locked down to any bias from a particular model. It also potentially opens or paves the way for multiple models to be referenced at once. A close second (in conjunction with the model being swappable) is the vector store being swappable as well. Having these two be likely to change components from the get-go ensured that this project was a framework rather than a script bound to specific models and vector stores that I used to put it together.

### Mechanism

1. **Model.** Everything talks to a `ModelClient` interface that is abstracted from the model itself (through `chat`, and `embed`). The `OllamaModelClient` is one example of this implementation. The model name lives in the config, meaning a model edit is just a config edit. The test suite swaps in a stub `ModelClient` with fake embeddings and no reference to `Ollama` in place of the real one. The test demonstrated that the memory store couldn't tell the difference and behaved identically.
2. **Vector Store.** A similar concept in that the vector store is interacted with via a `VectorStore` interface (through `add`, `query`), and `ChromaVectorStore` is just one example of this implementation. A different backend (such as a server-based store) could slot in by implementing the same contract, and it doesn't have to touch the retrieval or agent code.
3. **UI / App.** The app is dependent on the AI framework and not the reverse. The CLI is just the first consumer. Future implementations via the web or a voice frontend could sit on the same AI framework and be used as well. This speaks to the overall modularity of the framework, as the UI is also easily replaceable.
4. **Tools.** Each tool is like an arm of the AI framework. Tools must register into the `ToolRegistry`, and that doesn't modify the agent loop.
5. **Orchestrator (agent loop).** This is intentionally last because it's a part of everything else. An upgrade here would mean that something at the core needed to be updated for the entire AI framework, and likely other parts of the AI framework had to be updated in conjunction with the updates at this level.

There is definitely a future state here should it be pursued. Add new models on the backend using the same `ModelClient` relatively easy. Expanding into alternative vector stores (perhaps ones that are server based, use SQL, maybe distributed?) is quite possible, and additional applications can be added to the AI framework as well, enabling remote interaction via web-interface UI, or perhaps voice-control on the front-end. This was my goal: to be able to produce a framework that can be easily improved upon or added to.
