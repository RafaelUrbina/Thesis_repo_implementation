For your problem, I think the strongest architecture is:

**Geo dataset → vocabulary/ontology → patent text extraction → causal relation extraction → technical/environmental pairing → risk mechanism → topic modelling as a secondary discovery tool**

You are trying to discover **mechanisms by which environmental/spatial conditions can contribute to infrastructure failure**.

### 1. Your central object should be a causal chain

For example, from the patents you've shown:

> heavy rainfall → runoff → sediment accumulation → impeded flow → blockage → storm-drain failure

or:

> freezing conditions → extreme cold → mechanical/electrical components fail → water distribution problem

or:

> soil expansion/contraction → terrain movement → pipeline damage → leakage

This is much more informative for your geospatial application than a sentiment score.

I'd therefore structure each extracted patent relationship approximately as:

**Environment variable → causal mechanism → technical system component → failure mode**

For example:

| Environment          | Mechanism       | Technical system | Failure  |
| -------------------- | --------------- | ---------------- | -------- |
| heavy rainfall       | runoff          | storm drain      | blockage |
| freezing temperature | freezing        | water pipe       | failure  |
| soil expansion       | ground movement | pipeline         | damage   |
| landslide            | ground movement | pipeline         | rupture  |
| flooding             | inundation      | water network    | failure  |

That becomes the bridge between your GIS layers and the patent corpus.

---

## 2. Use your vocabulary as a retrieval layer first

Your current idea:

> "Get your vocabulary, take all appearances in documents, take all phrases that contain these phrases..."

is good.

But I would make it a **multi-stage extraction pipeline**, rather than immediately throwing the resulting text into LDA/BERTopic.

### Stage A — vocabulary matching

Search the full patent text for:

**A. Failure terms**

* leak
* rupture
* blockage
* damage
* failure
* etc.

**B. Causal terms**

* caused by
* leads to
* resulting from
* contributes to
* etc.

**C. Environmental variables**

* rainfall
* temperature
* flooding
* soil moisture
* landslide
* erosion
* freezing
* etc.

**D. Technical-system terms**

This fourth vocabulary is extremely important and is currently somewhat implicit in your project.

Examples:

* pipeline
* pipe
* valve
* pump
* storm drain
* water network
* conduit
* tank
* aquifer
* sensor
* reservoir

You need this because otherwise:

> "heavy rainfall causes flooding"

is interesting environmentally, but doesn't necessarily tell you anything about **pipeline risk**.

```text
ENVIRONMENT / CONTEXT
        │
        │
        ▼
     CAUSE
        │
        ▼
   MECHANISM
        │
        ▼
     EFFECT
        │
        ▼
    FAILURE
        │
        ▼
   CONSEQUENCE
        │
        ▼
TECHNICAL SOLUTION
```

But individual patents may only give you fragments:

```text
Patent A:
rainfall → flooding

Patent B:
flooding → soil movement

Patent C:
soil movement → pipeline deformation

Patent D:
pipeline deformation → pipeline failure

Patent E:
pipeline failure → water loss

Patent F:
pipeline deformation → monitoring system
```

Your objective should be to **join these pieces after extraction**.

Don't build one graph per patent

This is the biggest change I'd make to the previous code.

Instead of:

```text
Patent 123
   └── causal graph
```

build:

```text
                 ┌── Patent A
                 │
rainfall → flooding
                 │
                 ├── Patent B
                 ↓
            soil movement
                 │
                 ├── Patent C
                 ↓
        pipeline deformation
                 │
                 ├── Patent D
                 ↓
         pipeline failure
```

Every edge retains its **patent evidence**.

So the graph becomes a **corpus-level graph**.

---

# 3. Don't just extract the sentence — extract the surrounding context

I would initially take something like:

**±1–3 sentences around each matched term**

or perhaps a token window such as:

**±50–100 tokens**

around the match.

For example, if you find:

> "heavy rains"

don't just store `"heavy rains"`.

Store something like:

> [context window]
> "Rocks, gravel, sand, mud... are carried into storm drain conduits by storm runoff that is high in volume..."

Then identify:

**Variable:** storm/rainfall
**Mechanism:** runoff
**Technical system:** storm drain conduit
**Failure:** sediment accumulation / impeded flow

This gives your LLM much more information to work with.

---

# 4. Then perform causal relation extraction

This is probably the most important part of your project.

Instead of merely asking:

> "Does this patent mention flooding?"

ask:

> "Does this passage establish a relationship between an environmental condition and a technical-system condition?"

For example:

**Input**

> "Storm runoff is high in volume. Once the runoff volume decreases, material carried in the runoff settles and becomes sediment..."

You could extract:

```text
cause/condition: storm runoff
mechanism: material transport and deposition
technical object: storm drain conduit
failure state: sediment accumulation
effect: impeded flow
```

And importantly, preserve the **actual patent causal expression**:

> "becomes"

rather than immediately replacing it with a generic "causes."

This fits perfectly with your idea of having:

> a very specific connection using the verb used by the patent

and

> a general abstract relationship.

---

# 5. I would actually create partial causal graphs

This is a very good idea in your project.

### Level 1 — Abstract causal graph

Normalize it:

> precipitation → hydrological process → sediment deposition → flow restriction → infrastructure failure

This is your **ontology graph**.

That gives you both:

**Patent evidence**

and

**generalizable environmental-risk knowledge.**

I'd create **four retrieval buckets**.

### A. Context → Cause

```text
VARIABLE + CAUSAL
```

Examples:

```text
rainfall → erosion
temperature → freezing
soil moisture → soil movement
```

### B. Cause → Effect

```text
CAUSAL + FAILURE
```

Examples:

```text
erosion → pipeline damage
flooding → rupture
freezing → pipe failure
```

### C. Effect → Technical

```text
FAILURE + TECHNICAL
```

Examples:

```text
pipeline damage → monitoring system
leakage → acoustic sensor
rupture → pressure monitoring
```

### D. Technical → mitigation/solution

```text
TECHNICAL + CAUSAL
```

Examples:

```text
sensor detects leakage
monitoring system identifies damage
valve prevents freezing
```

You can then connect these separately extracted relationships.


---

# 6. Your "variable list" should become an ontology, not just a keyword list

This is an important evolution of your project.

For example:

```text
PRECIPITATION
 ├── rainfall
 ├── heavy rain
 ├── storm
 ├── storm event
 └── precipitation

TEMPERATURE
 ├── low temperature
 ├── freezing conditions
 ├── extreme cold
 └── freezing

GROUND MOVEMENT
 ├── landslide
 ├── terrain movement
 ├── soil movement
 ├── subsidence
 └── ground deformation
```

And then connect these to your actual GIS variables.

For example:

**PRECIPITATION**

→ `cf_pluviometri`

**TEMPERATURE**

→ `termometri_stations_firenze`

**GROUND MOVEMENT**

→ `frane_poly_toscana_opendata`
→ `velocita_movimento_nel_piano`
→ `celle_soli_PS_discendenti`
→ `celle_soli_PS_ascendenti`

**SOIL HYDROLOGY**

→ `ksat_30`
→ `ksat_150`
→ `awc`
→ `drenag`
→ `gi`

**FLOODING**

→ `hph...`
→ `mph...`
→ `lph...`
→ `inondaz`

This gives you:

**Patent vocabulary → conceptual variable → GIS variable**

which is much more powerful than simply searching for identical words.

---

# 7. Your final graph becomes a knowledge graph

I'd structure each extracted relation approximately like:

```text
source
source_type
relation
target
target_type
patent_id
sentence
context
confidence
```

Example:

```text
rainfall
VARIABLE
CAUSES
erosion
MECHANISM
US1234567
"Heavy rainfall can cause soil erosion..."
0.92
```

Another patent:

```text
erosion
MECHANISM
CAUSES
pipeline damage
FAILURE
US7654321
"Soil erosion can damage buried pipelines..."
0.88
```

Then you have:

```text
rainfall
   ↓
erosion
   ↓
pipeline damage
```

even though **no patent ever contained the entire chain**.

# 8. Then use topic modelling — but later

I **would use BERTopic**, but not as your first analytical method.

Your proposed idea is:

> vocabulary → occurrences → phrases → topic modelling

That's useful, but topic modelling answers a somewhat different question.

It can tell you:

> "What groups of concepts tend to occur together in this patent corpus?"

It does **not reliably tell you:**

> "Does rainfall cause pipeline failure?"

That's why I would use it as a **discovery layer**.

For example, BERTopic might discover a topic containing:

> rainfall, runoff, sediment, conduit, storm, flow, blockage

That is extremely useful.

You can then inspect that topic and potentially discover a new conceptual category:

**Stormwater → sediment transport → blockage**

You can subsequently add those concepts to your ontology.

So I would use:

### Causal extraction = primary analytical method

### BERTopic = discovery/exploration method

---

# 9. Your eventual database could look like this

This is where I think your project becomes really interesting.

Each patent passage produces a structured record:

| Field                | Example                                                |
| -------------------- | ------------------------------------------------------ |
| Patent               | US 10,927,539                                          |
| Environment variable | heavy rainfall                                         |
| GIS concept          | precipitation                                          |
| GIS layer            | `cf_pluviometri`                                       |
| Causal expression    | carried into                                           |
| Mechanism            | sediment transport                                     |
| Technical object     | storm drain conduit                                    |
| Failure term         | accumulated sediment                                   |
| Consequence          | impeded flow                                           |
| Specific relation    | runoff → carries sediment → conduit                    |
| Abstract relation    | precipitation → sediment deposition → flow restriction |
| Evidence text        | original patent sentence                               |
| Confidence           | 0.91                                                   |

Now you can eventually query:

> **Which patents describe mechanisms connecting rainfall to water infrastructure failure?**

or:

> **Which environmental variables have the strongest patent evidence for causing pipeline failures?**

or:

> **What failure modes are associated with soil movement?**

That's much more powerful than a simple keyword search.

---

# 9. Then your H3 hexagon grid becomes the final integration layer

This is where your existing GIS dataset becomes valuable.

Conceptually:

```text
PATENTS
   ↓
Vocabulary
   ↓
Environmental concepts
   ↓
Causal mechanisms
   ↓
Failure modes
   ↓
GIS variables
   ↓
H3 hexagon
   ↓
Risk features
```

For each H3 cell you could eventually have something like:

```text
H3 cell
│
├── rainfall
├── soil moisture
├── hydraulic conductivity
├── landslide susceptibility
├── erosion
├── flood probability
├── soil movement
├── temperature
├── drainage capacity
│
├── patent-supported causal mechanisms
│   ├── rainfall → runoff
│   ├── runoff → erosion
│   ├── erosion → exposure
│   └── soil movement → pipeline damage
│
└── potential failure modes
    ├── leakage
    ├── rupture
    ├── blockage
    └── structural damage
```

Then you have something resembling a **spatial infrastructure-risk ontology**.

---

architecture

```text
                PATENT CORPUS
                     │
                     ▼
             ┌───────────────┐
             │ #2 RETRIEVAL  │
             └───────┬───────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Variable      Failure      Technical
   contexts      contexts      contexts
        │            │            │
        └────────────┼────────────┘
                     ▼
             #3 CONTEXT WINDOW
                     │
                     ▼
             #4 RELATION EXTRACTION
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
      causal       causal       detection/
      relations    relations    mitigation
        │            │             │
        └────────────┼─────────────┘
                     ▼
             #5 GLOBAL GRAPH
                     │
                     ▼
           CORPUS CAUSAL NETWORK
                     │
                     ▼
                H3 GRID
```

The **global graph** is the important part.

---

## My recommended order

I would therefore change your workflow slightly:

**1. Build vocabulary**

Failure + causal + environmental + technical-system terms.

↓

**2. Expand vocabulary**

Wikipedia, engineering glossaries, FMEA, pipeline standards, environmental ontologies, LLM expansion.

Create Technical system terms.

Create an ontology for Variable terms, Causal to generic terms, Failure to generic temrs.

↓

**3. Retrieve patent passages**

Find occurrences and surrounding context.

↓

**4. Extract entities**

`environment → mechanism → technical system → failure`

↓

**5. Extract causal relationships**

Preserve the actual causal verb/expression from the patent.

↓

**6. Normalize relationships**

Create your generalized/abstract relationship.

↓

**7. Map environmental concepts → GIS layers**

For example:

`rainfall → precipitation → cf_pluviometri`

↓

**8. Map everything to H3**

Calculate the spatial environmental features.

↓

**9. BERTopic**

Use topic modelling on the extracted patent passages to discover **new mechanisms and clusters you didn't anticipate**.

↓

**10. Ontology/risk taxonomy**

Turn the discovered relationships into a structured taxonomy.

---

### One particularly important point

I would **not make the patent the starting point of the final risk score**.

Instead, let patents provide **evidence of plausible mechanisms**.

Your GIS data tells you:

> *Where and how strongly does the environmental condition exist?*

Your patent corpus tells you:

> *What mechanisms and failure modes could plausibly connect that condition to the technical system?*

So the final conceptual model becomes:

**Spatial exposure × causal mechanism × infrastructure susceptibility → potential failure risk**

That separation will make your methodology much more defensible academically.

And yes: **start pairing the patent full text with your geospatial variables now**, but start with **causal/relation extraction rather than sentiment or topic modelling**. Use BERTopic afterward as a discovery mechanism to find concepts and relationships your manually constructed vocabulary missed.

A useful architecture for your dataset would ultimately be:

                    ┌────────────────────┐
                    │   PATENT CORPUS    │
                    └─────────┬──────────┘
                              │
                       TERM RETRIEVAL
                              │
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
        VARIABLES          FAILURES        TECHNICAL
             │                │                │
             └────────────────┼────────────────┘
                              ↓
                     CONTEXT WINDOWS
                              ↓
                    CAUSAL CANDIDATES
                              ↓
                 CAUSAL RELATION CLASSIFIER
                              ↓
                    ┌─────────────────┐
                    │ EVIDENCE GRAPH  │
                    └────────┬────────┘
                             │
               ┌─────────────┼──────────────┐
               ↓             ↓              ↓
          variable→      failure→       failure→
           failure        failure        technical
               │             │              │
               └─────────────┼──────────────┘
                             ↓
                  CROSS-PATENT GRAPH
                             ↓
                 ENVIRONMENTAL RISK CHAIN
                             ↓
              H3 HEX-GRID RISK AGGREGATION