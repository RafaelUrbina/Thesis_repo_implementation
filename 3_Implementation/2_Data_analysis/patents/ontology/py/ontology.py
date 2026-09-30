"""Python script designed to automatically generate a Semantic Web Knowledge Graph
and Ontology for pipeline engineering, focusing on pipeline assets, failure
modes, and environmental & social risk variables.

Hierarchy per risk variable term: Concept Scheme -> Variable Classification
(ex:VariableClassification)
  -> Lemmatized Concept (skos:Concept) -> Original Surface Form
  (ex:SurfaceForm)
"""

import os
import re
import sys
from pathlib import Path

# Load spaCy for lemmatization
import spacy

nlp = spacy.load("en_core_web_sm")

# Ensure repository root is on sys.path by locating 'Utils' directory upwards
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "Utils").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
from Utils.paths import ONTOLOGY_LISTS_PATH, ONTOLOGY_OUTPUT_PATH

# Kept domain synonyms with length >= 4 to avoid 3-letter collisions
DOMAIN_SYNONYMS = {
    "inline inspection": ["smart pigging"],
    "supervisory control and data acquisition": ["scada"],
    "internal corrosion direct assessment": ["icda"],
    "external corrosion direct assessment": ["ecda"],
    "pressure relief valve": ["relief valve"],
    "water hammer": ["hydraulic shock"],
    "failure mode and effects analysis": ["fmea"],
    "hazard and operability study": ["hazop"],
}


def is_common_english_word(text: str) -> bool:
    """Checks if a string corresponds to a standard English word.
    
    Used to discard automatically generated initialisms that accidentally spell 
    out dictionary words (e.g., 'hand') rather than true technical abbreviations.
    """
    clean_text = text.lower()
    
    # 1. Reject if it is a stop word in spaCy
    if nlp.vocab[clean_text].is_stop:
        return True
        
    # 2. Check if the word is recognized as a valid word in spaCy's lookups/lexicon
    token = nlp.vocab[clean_text]
    
    # In en_core_web_sm, common dictionary words have rank < 200000 
    # or exist in the lexeme store with valid morphology/prob
    return not token.is_oov or token.prob != 0


def get_lemma(term: str) -> str:
    """Frames the term in a full sentence to supply POS context to spaCy."""
    clean_term = term.strip().lower()

    # Wrap in a simple English clause
    doc = nlp(f"This is a {clean_term}.")

    # Extract tokens belonging ONLY to the term (ignoring "This", "is", "a", ".")
    term_tokens = doc[3:-1]

    lemmatized = " ".join([token.lemma_ for token in term_tokens])
    # Remove extra whitespace around hyphens caused by spaCy tokenization
    return re.sub(r"\s*-\s*", "-", lemmatized)


def parse_term_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Skipping.")
        return []
    terms_data = []
    seen = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.lower() in seen:
                continue
            seen.add(clean_line.lower())
            alts = []
            pref = clean_line
            match = re.match(r"^(.*?)\s*\((.*?)\)$", clean_line)
            if match:
                pref = match.group(1).strip()
                alt = match.group(2).strip()
                # Ignore 3-letter acronyms from explicitly parenthesized alternatives
                if alt and len(alt) > 3:
                    alts.append(alt)
            pref_lower = pref.lower()
            if pref_lower in DOMAIN_SYNONYMS:
                for syn in DOMAIN_SYNONYMS[pref_lower]:
                    if syn not in alts and syn != pref_lower and len(syn) > 3:
                        alts.append(syn)
            words = pref.split()
            if len(words) >= 3 and not alts:
                initialism = "".join(
                    w[0] for w in words if w.isalnum()
                ).lower()
                # Filter out initialisms <= 3 chars or those that spell valid English words (e.g., 'hand')
                if (
                    len(initialism) > 3
                    and initialism != pref_lower
                    and not is_common_english_word(initialism)
                ):
                    alts.append(initialism)
            terms_data.append({"pref": pref, "alts": alts})
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
    causal_file=ONTOLOGY_LISTS_PATH / "causal_list.txt",
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
    g.add((
        ont_uri,
        RDFS.comment,
        Literal(
            "Unified Knowledge Graph of Pipeline Assets, Failure Modes, Causal"
            " Relations, and Environmental/Social Risk Variables.",
            lang="en",
        ),
    ))

    # Concept Schemes (4 Pillars)
    schemes = {
        "technical": EX.TechnicalComponentScheme,
        "failure": EX.FailureModeScheme,
        "risk": EX.EnvironmentalSocialRiskScheme,
        "causal": EX.CausalRelationalScheme,
    }

    for key, uri in schemes.items():
        g.add((uri, RDF.type, SKOS.ConceptScheme))
        g.add((
            uri,
            SKOS.prefLabel,
            Literal(f"Pipeline {key.title()} Concept Scheme", lang="en"),
        ))

    def add_category_root(
        scheme,
        category_id,
        pref_label,
        broader_uri=None,
        is_variable_class=False,
    ):
        """Creates category nodes within a scheme."""
        uri = EX[category_id]
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, SKOS.inScheme, scheme))
        g.add((uri, SKOS.prefLabel, Literal(pref_label, lang="en")))

        if is_variable_class:
            g.add((uri, RDF.type, EX.VariableClassification))
            g.add((scheme, SKOS.hasTopConcept, uri))

        if broader_uri:
            g.add((uri, SKOS.broader, broader_uri))
            g.add((broader_uri, SKOS.narrower, uri))
            g.add((uri, RDFS.subClassOf, broader_uri))
        return uri

    def add_term_nodes(scheme, pref_label, alt_labels=None, broader_uri=None):
        """Creates structure: Lemma Concept -> Surface Form Instance."""
        lemmatized_str = get_lemma(pref_label)

        lemma_id = f"Lemma_{slugify(lemmatized_str)}"
        surface_id = f"Term_{slugify(pref_label)}"

        lemma_uri = EX[lemma_id]
        surface_uri = EX[surface_id]

        # --- Layer 1: Scheme / Class -> Lemmatized Concept ---
        g.add((lemma_uri, RDF.type, SKOS.Concept))
        g.add((lemma_uri, RDF.type, OWL.Class))
        g.add((lemma_uri, SKOS.inScheme, scheme))
        g.add((lemma_uri, SKOS.prefLabel, Literal(lemmatized_str, lang="en")))

        if broader_uri:
            g.add((lemma_uri, SKOS.broader, broader_uri))
            g.add((broader_uri, SKOS.narrower, lemma_uri))
            g.add((lemma_uri, RDFS.subClassOf, broader_uri))

        # --- Layer 2: Lemmatized Concept -> Original Surface Term ---
        g.add((surface_uri, RDF.type, EX.SurfaceForm))
        g.add((surface_uri, EX.hasLemma, lemma_uri))
        g.add((surface_uri, RDFS.label, Literal(pref_label, lang="en")))

        # Add synonyms and alternative forms to the surface term & lemma
        if alt_labels:
            for alt in alt_labels:
                g.add((surface_uri, SKOS.altLabel, Literal(alt, lang="en")))
                g.add((lemma_uri, SKOS.altLabel, Literal(alt, lang="en")))

        return lemma_uri

    # 3. Process Technical Terms
    tech_terms = parse_term_file(tech_file)
    if tech_terms:
        root_tech = add_category_root(
            schemes["technical"],
            "TechnicalTerm",
            "technical term",
        )
        print(f"Ingesting {len(tech_terms)} technical components...")
        for item in tech_terms:
            if item["pref"]:
                add_term_nodes(
                    schemes["technical"],
                    item["pref"],
                    alt_labels=item["alts"],
                    broader_uri=root_tech,
                )

    # 4. Process Failure Mode Terms
    failure_terms = parse_term_file(failure_file)
    if failure_terms:
        root_fail = add_category_root(
            schemes["failure"], "FailureTerm", "failure term"
        )

        corr_fail = add_category_root(
            schemes["failure"],
            "CorrosionAndElectrochemicalFailure",
            "corrosion and electrochemical failure",
            broader_uri=root_fail,
        )
        mech_fail = add_category_root(
            schemes["failure"],
            "MechanicalAndStructuralFailure",
            "mechanical and structural failure",
            broader_uri=root_fail,
        )
        geo_fail = add_category_root(
            schemes["failure"],
            "GeotechnicalAndGroundFailure",
            "geotechnical and ground failure",
            broader_uri=root_fail,
        )
        hydr_fail = add_category_root(
            schemes["failure"],
            "HydraulicAndOperationalSurgeFailure",
            "hydraulic and operational surge failure",
            broader_uri=root_fail,
        )
        third_fail = add_category_root(
            schemes["failure"],
            "ThirdPartyAndExternalInterference",
            "third party and external interference",
            broader_uri=root_fail,
        )

        print(f"Ingesting {len(failure_terms)} failure modes...")
        for item in failure_terms:
            pref = item["pref"]
            if not pref:
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

            add_term_nodes(
                schemes["failure"],
                pref,
                alt_labels=item["alts"],
                broader_uri=parent,
            )

    # 5. Process Risk Variable Terms (4-Tier Hierarchy)
    risk_terms = parse_term_file(risk_file)
    if risk_terms:
        # Define the 8 Variable Classification Categories (Tier 2)
        class_meteo = add_category_root(
            schemes["risk"],
            "Class_MeteorologicalMonitoring",
            "Meteorological & Hydrological Monitoring",
            is_variable_class=True,
        )
        class_hydro = add_category_root(
            schemes["risk"],
            "Class_HydrologicalSettingsAndFloodRisk",
            "Hydrological Settings & Flood Risk",
            is_variable_class=True,
        )
        class_geotech = add_category_root(
            schemes["risk"],
            "Class_GeotechnicalDynamics",
            "Geotechnical Dynamics & Landslide Hazards",
            is_variable_class=True,
        )
        class_pedo = add_category_root(
            schemes["risk"],
            "Class_PedologicalProperties",
            "Pedological Soil Properties & Agricultural Capacity",
            is_variable_class=True,
        )
        class_topo_clim = add_category_root(
            schemes["risk"],
            "Class_TopographicAndClimaticIndicators",
            "Topographic & Climatic Indicators",
            is_variable_class=True,
        )
        class_built_env = add_category_root(
            schemes["risk"],
            "Class_BuiltEnvironmentAndLandUse",
            "Built Environment, Land Use, & Human Interventions",
            is_variable_class=True,
        )
        class_landscape = add_category_root(
            schemes["risk"],
            "Class_LandscapeRegionalization",
            "Landscape Regionalization",
            is_variable_class=True,
        )
        class_socio = add_category_root(
            schemes["risk"],
            "Class_SocioEconomicIndicators",
            "Socio-Economic Indicators, Vulnerability, & Municipal Metabolism",
            is_variable_class=True,
        )

        print(
            f"Ingesting {len(risk_terms)} environmental/social risk"
            " variables into 4-tier hierarchy..."
        )
        for item in risk_terms:
            pref = item["pref"]
            if not pref:
                continue

            plow = pref.lower()
            parent = class_socio  # Fallback classification

            # Specific keyword matching rules for the 8 classification categories
            if any(
                k in plow
                for k in [
                    "gauge",
                    "real-time discharge",
                    "station",
                    "sensor",
                    "monitoring",
                ]
            ):
                parent = class_meteo
            elif any(
                k in plow
                for k in [
                    "flood",
                    "river",
                    "waterway",
                    "culvert",
                    "stormwater",
                    "hydraulic risk",
                    "inundation",
                    "hydrographic",
                    "run-off",
                    "runoff",
                ]
            ):
                parent = class_hydro
            elif any(
                k in plow
                for k in [
                    "landslide",
                    "mass movement",
                    "mass wasting",
                    "seismic",
                    "earthquake",
                    "ground deformation",
                    "downward deformation",
                    "vertical deformation",
                    "slope movement",
                ]
            ):
                parent = class_geotech
            elif any(
                k in plow
                for k in [
                    "hydrologic soil group",
                    "ksat",
                    "hydraulic conductivity",
                    "salinity",
                    "soil depth",
                    "rooting depth",
                    "internal drainage",
                    "electrical conductivity",
                    "pedological",
                ]
            ):
                parent = class_pedo
            elif any(
                k in plow
                for k in [
                    "elevation",
                    "drought",
                    "climate interference",
                    "altitudinal",
                    "moisture deficit",
                    "topographic",
                ]
            ):
                parent = class_topo_clim
            elif any(
                k in plow
                for k in [
                    "building footprint",
                    "building density",
                    "architectural density",
                    "remediation cost",
                    "repair cost",
                    "damage repair",
                    "built environment",
                ]
            ):
                parent = class_built_env
            elif any(
                k in plow
                for k in [
                    "landscape unit",
                    "unita di paesaggio",
                    "soil region",
                    "landscape system",
                    "subsystem",
                    "regionalization",
                ]
            ):
                parent = class_landscape
            elif any(
                k in plow
                for k in [
                    "population",
                    "income",
                    "wealth",
                    "commuter",
                    "water metabolism",
                    "taxable",
                    "pension",
                    "economic magnetism",
                    "daytime population",
                    "demographic",
                ]
            ):
                parent = class_socio

            add_term_nodes(
                schemes["risk"],
                pref,
                alt_labels=item["alts"],
                broader_uri=parent,
            )

    # 6. Process Causal Terms
    causal_terms = parse_term_file(causal_file)
    if causal_terms:
        root_causal = add_category_root(
            schemes["causal"],
            "CausalTerm",
            "causal term",
        )

        dir_causal = add_category_root(
            schemes["causal"],
            "DirectCausationRelation",
            "direct causation relation",
            broader_uri=root_causal,
        )
        cond_causal = add_category_root(
            schemes["causal"],
            "ConditionalAndContributingRelation",
            "conditional and contributing relation",
            broader_uri=root_causal,
        )
        corr_causal = add_category_root(
            schemes["causal"],
            "CorrelationAndAssociationRelation",
            "correlation and association relation",
            broader_uri=root_causal,
        )
        attr_causal = add_category_root(
            schemes["causal"],
            "AttributionAndDerivationRelation",
            "attribution and derivation relation",
            broader_uri=root_causal,
        )

        print(f"Ingesting {len(causal_terms)} causal terms...")
        for item in causal_terms:
            pref = item["pref"]
            if not pref:
                continue

            plow = pref.lower()
            parent = root_causal

            if any(
                k in plow
                for k in [
                    "cause",
                    "lead",
                    "result",
                    "produce",
                    "give rise",
                    "break",
                    "create",
                ]
            ):
                parent = dir_causal
            elif any(
                k in plow
                for k in [
                    "contribute",
                    "trigger",
                    "induce",
                    "exacerbated",
                    "allow",
                    "require",
                    "prevent",
                    "exceed",
                    "impos",
                ]
            ):
                parent = cond_causal
            elif any(
                k in plow
                for k in [
                    "associate",
                    "correlate",
                    "link",
                    "correspond",
                    "common",
                    "depend",
                ]
            ):
                parent = corr_causal
            elif any(
                k in plow
                for k in [
                    "due to",
                    "attribute",
                    "derive",
                    "base",
                    "originate",
                    "owing",
                    "stem",
                ]
            ):
                parent = attr_causal

            add_term_nodes(
                schemes["causal"],
                pref,
                alt_labels=item["alts"],
                broader_uri=parent,
            )

    # 7. Export Graph
    g.serialize(
        destination=ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.ttl",
        format="turtle",
    )
    g.serialize(
        destination=ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.rdf",
        format="xml",
        base="http://www.semanticweb.org/thesis/pipeline-risk#",
    )

    print(f"\nGraph Generation Complete! Total RDF Triples: {len(g)}")


if __name__ == "__main__":
    build_unified_knowledge_graph()