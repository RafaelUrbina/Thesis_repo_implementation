""" Python script designed to automatically generate a __Semantic Web Knowledge Graph and Ontology__ 
for pipeline engineering, focusing on pipeline assets, failure modes, and environmental & social risk variables.

It leverages the __`rdflib`__ library to construct a formal Resource Description Framework (RDF) knowledge graph 
using established semantic web standards (`SKOS`, `OWL`, `RDFS`), and serializes it into standard 
formats (`.ttl` Turtle and `.rdf` XML).

Here is a detailed breakdown of how the script works, its core functions, and its architecture:

---

### 1. Core Helper Functions

- __`parse_term_list(terms_raw)`__:

  - Cleans raw text strings by stripping whitespace and filtering out case-insensitive duplicates.
  - Uses regular expressions (`re.match`) to extract parenthetical acronyms or synonyms 
  (e.g., converting `"topographic wetness index (twi)"` into a preferred label `"topographic wetness index"` and an 
  alternative label `"twi"`). These are later mapped to `skos:altLabel`.

- __`slugify(text)`__:

  - Converts descriptive phrase labels into clean, strict PascalCase URI identifiers 
  (e.g., `"Soil Draining Capability"` becomes `"SoilDrainingCapability"`).

 """
import os
import re
from pathlib import Path
import sys

# Ensure repository root is on sys.path by locating 'Utils' directory upwards
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "Utils").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import ONTOLOGY_LISTS_PATH
from Utils.paths import ONTOLOGY_OUTPUT_PATH

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD


def parse_term_file(filepath):
    """Reads a text file line-by-line, cleans whitespace, removes exact duplicates,

    and extracts parenthetical acronyms/synonyms as SKOS altLabels.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File '{filepath}' not found. Skipping.")
        return []

    terms_data = []
    seen = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.lower() in seen:
                continue
            seen.add(clean_line.lower())

            # Extract parenthetical synonyms (e.g., "topographic wetness index (twi)")
            match = re.match(r"^(.*?)\s*\((.*?)\)$", clean_line)
            if match:
                pref = match.group(1).strip()
                alt = match.group(2).strip()
                terms_data.append({"pref": pref, "alts": [alt]})
            else:
                terms_data.append({"pref": clean_line, "alts": []})

    return terms_data


def slugify(text):
    """Converts a phrase into a clean PascalCase URI identifier."""
    cleaned = re.sub(r"[^\w\s-]", "", text)
    words = cleaned.title().split()
    return "".join(words)


def build_unified_knowledge_graph(
    tech_file=ONTOLOGY_LISTS_PATH / "technical_list.txt",
    failure_file=ONTOLOGY_LISTS_PATH / "failure_list.txt",
    risk_file=ONTOLOGY_LISTS_PATH / "variable_list.txt",
):
    g = Graph()
    EX = Namespace("http://www.semanticweb.org/thesis/pipeline-risk#")

    # Bind Namespace Prefixes
    g.bind("ex", EX)
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # Base Ontology Declaration
    ont_uri = URIRef("http://www.semanticweb.org/thesis/pipeline-risk")
    g.add((ont_uri, RDF.type, OWL.Ontology))
    g.add(
        (
            ont_uri,
            RDFS.comment,
            Literal(
                "Unified Knowledge Graph of Pipeline Assets, Failure Modes, and Environmental/Social Risk Variables.",
                lang="en",
            ),
        )
    )

    # 2. Concept Schemes
    schemes = {
        "technical": EX.TechnicalComponentScheme,
        "failure": EX.FailureModeScheme,
        "risk": EX.EnvironmentalSocialRiskScheme,
    }

    for key, uri in schemes.items():
        g.add((uri, RDF.type, SKOS.ConceptScheme))
        g.add(
            (
                uri,
                SKOS.prefLabel,
                Literal(f"Pipeline {key.title()} Concept Scheme", lang="en"),
            )
        )

    def add_concept_node(
        scheme, concept_id, pref_label, alt_labels=None, broader_uri=None
    ):
        uri = EX[concept_id]
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, SKOS.inScheme, scheme))
        g.add((uri, SKOS.prefLabel, Literal(pref_label.lower(), lang="en")))

        if broader_uri:
            g.add((uri, SKOS.broader, broader_uri))
            g.add((broader_uri, SKOS.narrower, uri))
            g.add((uri, RDFS.subClassOf, broader_uri))

        if alt_labels:
            for alt in alt_labels:
                g.add((uri, SKOS.altLabel, Literal(alt.lower(), lang="en")))
        return uri

    # 3. Process Technical Terms
    tech_terms = parse_term_file(tech_file)
    if tech_terms:
        root_tech = add_concept_node(
            schemes["technical"],
            "TechnicalSolutionComponent",
            "Technical Solution and Component",
        )
        print(f"Ingesting {len(tech_terms)} technical components...")
        for item in tech_terms:
            cid = slugify(item["pref"])
            if cid:
                add_concept_node(
                    schemes["technical"],
                    cid,
                    item["pref"],
                    alt_labels=item["alts"],
                    broader_uri=root_tech,
                )

    # 4. Process Failure Mode Terms
    failure_terms = parse_term_file(failure_file)
    if failure_terms:
        root_fail = add_concept_node(
            schemes["failure"], "PipelineFailureMode", "Pipeline Failure Mode"
        )

        corr_fail = add_concept_node(
            schemes["failure"],
            "CorrosionAndElectrochemicalFailure",
            "Corrosion and Electrochemical Failure",
            broader_uri=root_fail,
        )
        mech_fail = add_concept_node(
            schemes["failure"],
            "MechanicalAndStructuralFailure",
            "Mechanical and Structural Failure",
            broader_uri=root_fail,
        )
        geo_fail = add_concept_node(
            schemes["failure"],
            "GeotechnicalAndGroundFailure",
            "Geotechnical and Ground Failure",
            broader_uri=root_fail,
        )
        hydr_fail = add_concept_node(
            schemes["failure"],
            "HydraulicAndOperationalSurgeFailure",
            "Hydraulic and Operational Surge Failure",
            broader_uri=root_fail,
        )
        third_fail = add_concept_node(
            schemes["failure"],
            "ThirdPartyAndExternalInterference",
            "Third Party and External Interference",
            broader_uri=root_fail,
        )

        print(f"Ingesting {len(failure_terms)} failure modes...")
        for item in failure_terms:
            pref = item["pref"]
            cid = slugify(pref)
            if not cid:
                continue

            plow = pref.lower()
            parent = root_fail
            if any(
                k in plow
                for k in [
                    "corrosion",
                    "pitting",
                    "rust",
                    "oxidation",
                    "cathodic",
                    "anodic",
                    "galvanic",
                    "hydrogen",
                    "microbial",
                    "biofouling",
                ]
            ):
                parent = corr_fail
            elif any(
                k in plow
                for k in [
                    "fatigue",
                    "buckling",
                    "dent",
                    "gouge",
                    "crack",
                    "wear",
                    "creep",
                    "deformation",
                    "rupture",
                    "fracture",
                    "weld",
                    "strain",
                ]
            ):
                parent = mech_fail
            elif any(
                k in plow
                for k in [
                    "soil",
                    "ground",
                    "subsidence",
                    "landslide",
                    "seismic",
                    "heave",
                    "liquefaction",
                    "frost",
                    "earthquake",
                ]
            ):
                parent = geo_fail
            elif any(
                k in plow
                for k in [
                    "hammer",
                    "surge",
                    "cavitation",
                    "pressure",
                    "flow",
                    "transient",
                    "hydraulic",
                ]
            ):
                parent = hydr_fail
            elif any(
                k in plow
                for k in [
                    "third-party",
                    "excavation",
                    "sabotage",
                    "hit",
                    "strike",
                    "interference",
                    "anchor",
                    "dredging",
                ]
            ):
                parent = third_fail

            add_concept_node(
                schemes["failure"],
                cid,
                pref,
                alt_labels=item["alts"],
                broader_uri=parent,
            )

    # 5. Process Risk Variable Terms
    risk_terms = parse_term_file(risk_file)
    if risk_terms:
        root_risk = add_concept_node(
            schemes["risk"],
            "EnvironmentalAndSocialRiskVariable",
            "Environmental and Social Risk Variable",
        )

        hydro_risk = add_concept_node(
            schemes["risk"],
            "HydrologicalAndInundationVariable",
            "Hydrological and Inundation Variable",
            broader_uri=root_risk,
        )
        geo_risk = add_concept_node(
            schemes["risk"],
            "GeomorphologicalAndGeotechnicalVariable",
            "Geomorphological and Geotechnical Variable",
            broader_uri=root_risk,
        )
        ped_risk = add_concept_node(
            schemes["risk"],
            "PedologicalAndSoilConditionVariable",
            "Pedological and Soil Condition Variable",
            broader_uri=root_risk,
        )
        clim_risk = add_concept_node(
            schemes["risk"],
            "ClimaticAndAtmosphericVariable",
            "Climatic and Atmospheric Variable",
            broader_uri=root_risk,
        )
        soc_risk = add_concept_node(
            schemes["risk"],
            "SocioEconomicAndUrbanSpatioVariable",
            "Socio-Economic and Urban Spatial Variable",
            broader_uri=root_risk,
        )

        print(f"Ingesting {len(risk_terms)} environmental/social risk variables...")
        for item in risk_terms:
            pref = item["pref"]
            cid = slugify(pref)
            if not cid:
                continue

            plow = pref.lower()
            parent = root_risk

            if any(
                k in plow
                for k in [
                    "flood",
                    "rain",
                    "water",
                    "river",
                    "stream",
                    "inundation",
                    "discharge",
                    "runoff",
                    "aquifer",
                    "lake",
                    "hydrol",
                    "hydrog",
                    "catchment",
                    "wetness",
                    "drainage",
                ]
            ):
                parent = hydro_risk
            elif any(
                k in plow
                for k in [
                    "landslide",
                    "slope",
                    "terrain",
                    "elevation",
                    "seismic",
                    "earthquake",
                    "fault",
                    "bedrock",
                    "rock",
                    "avalanche",
                    "subsidence",
                    "geol",
                    "geomorph",
                    "gradient",
                    "topographic",
                    "mass wasting",
                ]
            ):
                parent = geo_risk
            elif any(
                k in plow
                for k in [
                    "soil",
                    "clay",
                    "silt",
                    "sand",
                    "ksat",
                    "pedol",
                    "salinity",
                    "organic",
                    "cation",
                    "porosity",
                    "bulk density",
                    "horizon",
                    "suction",
                ]
            ):
                parent = ped_risk
            elif any(
                k in plow
                for k in [
                    "climate",
                    "temperature",
                    "wind",
                    "storm",
                    "ice",
                    "frost",
                    "fog",
                    "drought",
                    "atmospheric",
                    "humidity",
                    "weather",
                    "solar",
                ]
            ):
                parent = clim_risk
            elif any(
                k in plow
                for k in [
                    "urban",
                    "city",
                    "town",
                    "building",
                    "population",
                    "income",
                    "cadastral",
                    "road",
                    "demographic",
                    "tax",
                    "administrative",
                    "economic",
                    "spending",
                    "cost",
                    "vegetation",
                    "forest",
                    "land use",
                    "agricultural",
                ]
            ):
                parent = soc_risk

            add_concept_node(
                schemes["risk"],
                cid,
                pref,
                alt_labels=item["alts"],
                broader_uri=parent,
            )

    # 6. Export Graph
    g.serialize(destination=ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.ttl", format="turtle")
    g.serialize(destination=ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.rdf", format="xml")

    print(f"\nGraph Generation Complete! Total RDF Triples: {len(g)}")


if __name__ == "__main__":
    build_unified_knowledge_graph()