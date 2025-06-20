import json
import random
import math
import os
from typing import List, Dict, Any, Tuple, Union
# For Parquet conversion, you'd need:
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# --- Configuration ---
NUM_SAMPLES_PER_TYPE = 30  # Generate 30 samples for each task type
BASE_OUTPUT_FILENAME = "extended_seed_data_v3" # Base name for output files
OUTPUT_DIR = "generated_seed_data" # Directory to store output files

# --- Difficulty Levels & Heuristics ---
DIFFICULTY_LEVELS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "very_hard": 4,
    "expert": 5
}

# --- Data for Syllogisms (Expanded) ---
SYLLOGISM_SUBJECTS = ["cats", "dogs", "birds", "fish", "students", "teachers", "programmers", "AIs", "robots", "planets", "stars", "ideas", "concepts", "cars", "trees", "books", "cities"]
SYLLOGISM_PREDICATES = ["mammals", "animals", "mortals", "learners", "thinkers", "creatures", "entities", "physical_objects", "abstractions", "systems", "machines", "living_things", "constructs"]
SYLLOGISM_MIDDLE_TERMS = ["pets", "warm-blooded_beings", "intelligent_beings", "carbon-based_lifeforms", "complex_systems", "household_items", "celestial_bodies", "tools", "organisms"]

SYLLOGISM_TEMPLATES = [
    # Valid Forms
    {"id": "S_VALID_AAA-1", "form": "AAA-1", "format_str": "All {m} are {p}. All {s} are {m}. Therefore, all {s} are {p}.", "valid": True, "output_template": "Yes. This is a valid syllogism (Barbara): If all {m} are {p} and all {s} are {m}, then all {s} are {p}.", "reasoning_skill": "syllogistic_deduction", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"id": "S_VALID_EAE-1", "form": "EAE-1", "format_str": "No {m} are {p}. All {s} are {m}. Therefore, no {s} are {p}.", "valid": True, "output_template": "Yes. This is a valid syllogism (Celarent).", "reasoning_skill": "syllogistic_deduction", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"id": "S_VALID_AII-3", "form": "AII-3", "format_str": "All {m} are {p}. Some {m} are {s}. Therefore, some {s} are {p}.", "valid": True, "output_template": "Yes. This is a valid syllogism (Datisi).", "reasoning_skill": "syllogistic_deduction", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"id": "S_VALID_OAO-3", "form": "OAO-3", "format_str": "Some {m} are not {p}. All {m} are {s}. Therefore, some {s} are not {p}.", "valid": True, "output_template": "Yes. This is a valid syllogism (Bokardo).", "reasoning_skill": "syllogistic_deduction", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    # Invalid Forms
    {"id": "S_INVALID_UNDISTRIBUTED_MIDDLE", "form": "AAA-2 (Invalid)", "format_str": "All {p} are {m}. All {s} are {m}. Therefore, all {s} are {p}.", "valid": False, "output_template": "No. This syllogism is invalid due to the fallacy of the undistributed middle term.", "reasoning_skill": "fallacy_detection", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"id": "S_INVALID_ILLICIT_MAJOR", "form": "AEE-1 (Invalid)", "format_str": "All {m} are {p}. No {s} are {m}. Therefore, no {s} are {p}.", "valid": False, "output_template": "No. This syllogism is invalid due to the fallacy of the illicit major term.", "reasoning_skill": "fallacy_detection", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["hard"]},
    {"id": "S_INVALID_FALSE_PREMISE", "form": "AAA-1 (Valid Form, False Premise)", "format_str": "All {s} are {m}. All {m} are {p}. Therefore, all {s} are {p}.", "valid": False, "output_template": "No. While the form is valid, if a premise is false (e.g., '{false_premise}'), the conclusion isn't guaranteed. Assess premises first.", "reasoning_skill": "premise_evaluation", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["hard"]},
    {"id": "S_INVALID_AMBIGUOUS_SOME", "form": "III-1 (Invalid)", "format_str": "Some {m} are {p}. Some {s} are {m}. Therefore, some {s} are {p}.", "valid": False, "output_template": "No. This conclusion does not necessarily follow due to ambiguity of 'some' (fallacy of two particular premises).", "reasoning_skill": "quantifier_logic", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
]

# --- Data for Truth Assessment (Expanded) ---
TRUTH_ASSESSMENT_CLAIMS = [
    {"claim": "The Earth is flat.", "truth": False, "consensus": True, "certainty": 0.99, "source_quality": "low (disproven)", "reasoning_skill": "scientific_literacy", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"claim": "Water boils at 100°C at standard atmospheric pressure.", "truth": True, "consensus": True, "certainty": 1.0, "source_quality": "high (scientific fact)", "reasoning_skill": "scientific_literacy", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"claim": "Vaccines have been scientifically proven to cause autism in children.", "truth": False, "consensus": True, "certainty": 0.99, "source_quality": "very low (disproven, harmful misinformation)", "reasoning_skill": "evaluating_misinformation", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"claim": "Climate change is primarily caused by human activities, according to the IPCC.", "truth": True, "consensus": True, "certainty": 0.97, "source_quality": "high (IPCC consensus)", "reasoning_skill": "scientific_literacy", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"claim": "There is strong evidence suggesting liquid water may exist under the ice shell of Europa, a moon of Jupiter.", "truth": "Plausible", "consensus": True, "certainty": 0.75, "source_quality": "high (current scientific hypothesis)", "reasoning_skill": "scientific_inference", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"claim": "All online news sources present information with equal objectivity and factual accuracy.", "truth": False, "consensus": True, "certainty": 0.95, "source_quality": "varied (common misconception)", "reasoning_skill": "media_literacy", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"claim": "Artificial general intelligence (AGI) will be achieved and widely deployed by the year 2030.", "truth": "Highly Speculative", "consensus": False, "certainty": 0.1, "source_quality": "varied (expert opinions differ wildly, often sensationalized)", "reasoning_skill": "evaluating_speculative_claims", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["hard"]},
    {"claim": "Eating carrots grants superhuman night vision.", "truth": False, "consensus": True, "certainty": 0.90, "source_quality": "medium (myth based on WWII propaganda, Vitamin A is good for vision but won't give superpowers)", "reasoning_skill": "myth_debunking", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"claim": "The theory of evolution by natural selection is a well-substantiated scientific theory.", "truth": True, "consensus": True, "certainty": 0.99, "source_quality": "high (foundational biology)", "reasoning_skill": "scientific_literacy", "confounder_present": False, "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"claim": "Cold fusion has been reliably demonstrated and is a viable energy source.", "truth": False, "consensus": True, "certainty": 0.90, "source_quality": "low (initial claims not reproduced, fringe science)", "reasoning_skill": "evaluating_scientific_claims", "confounder_present": True, "base_difficulty": DIFFICULTY_LEVELS["hard"]},
]

# --- Data for Graph Construction & Inference (Expanded) ---
GRAPH_NODES_CONCEPTS = ["Fire", "Smoke", "Rain", "Flood", "Deforestation", "Soil Erosion", "Algorithm", "Data Structure", "CPU", "Memory", "Software Bug", "System Crash", "User Action", "Security Vulnerability", "Encryption", "Decryption", "Authentication", "Authorization", "Virus", "Malware", "Firewall", "Network Traffic", "DNS Resolution"]
GRAPH_EDGE_TYPES = ["causes", "is_caused_by", "depends_on", "enables", "prevents", "mitigates", "is_a_type_of", "part_of", "correlates_with", "implies", "requires", "blocks"]
GRAPH_STRUCTURE_TEMPLATES = [
    {"type": "simple_relation", "nodes": 2, "edges": 1, "snippet_template": "Represent the relationship: '{n1}' {e_type} '{n2}'.", "reasoning_skill": "relational_graph_construction", "base_difficulty": DIFFICULTY_LEVELS["easy"]},
    {"type": "chain_of_three", "nodes": 3, "edges": 2, "snippet_template": "Show that '{n1}' {e_type1} '{n2}', and '{n2}' {e_type2} '{n3}'.", "reasoning_skill": "dependency_chain_graphing", "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"type": "common_cause", "nodes": 3, "edges": 2, "snippet_template": "Illustrate that '{n1}' {e_type1} '{n2}', and '{n1}' also {e_type2} '{n3}'.", "reasoning_skill": "common_cause_graphing", "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"type": "common_effect", "nodes": 3, "edges": 2, "snippet_template": "Show that '{n1}' {e_type1} '{n3}', and '{n2}' also {e_type2} '{n3}'.", "reasoning_skill": "common_effect_graphing", "base_difficulty": DIFFICULTY_LEVELS["medium"]},
    {"type": "cyclic_three", "nodes": 3, "edges": 3, "snippet_template": "Illustrate a cycle: '{n1}' {e_type1} '{n2}', '{n2}' {e_type2} '{n3}', and '{n3}' {e_type3} '{n1}'.", "reasoning_skill": "cyclic_graph_representation", "base_difficulty": DIFFICULTY_LEVELS["hard"]},
]
GRAPH_INFERENCE_QUESTIONS = [
    {"graph_snippet_template": "Given: {n1} {e1} {n2}, and {n2} {e2} {n3}.", "question_template": "If {e1} and {e2} imply transitivity for these types of relations, does {n1} have an indirect relationship with {n3}?", "expected_answer_logic": lambda e1, e2: "Yes, an indirect relationship exists." if e1 in ["causes", "leads_to", "implies", "depends_on"] and e2 in ["causes", "leads_to", "implies", "depends_on"] else "Cannot be definitively concluded without more information on the nature of the relationships.", "reasoning_skill": "transitive_inference_graph", "base_difficulty_modifier": 1},
    {"graph_snippet_template": "Graph: {n1} {e1} {target}; {n2} {e2} {target}; {n3} {e3} {other_node}.", "question_template": "Which of the listed nodes directly {e_target_rel} '{target}'?", "expected_answer_logic": lambda n1, n2, target, e1, e2: f"Nodes '{n1}' (via {e1}) and '{n2}' (via {e2}) directly influence '{target}'.", "reasoning_skill": "identifying_direct_influences", "base_difficulty_modifier": 0},
    {"graph_snippet_template": "Consider: {n1} {e1} {n2}; {n2} {e2} {n3}; {n3} {e3} {n1}.", "question_template": "Does this graph structure contain a cycle?", "expected_answer_logic": lambda: "Yes, this structure describes a cycle.", "reasoning_skill": "cycle_detection", "base_difficulty_modifier": 0},
]

# --- Data for Logical Conflict (Expanded) ---
LOGICAL_CONFLICT_PREMISES = [
    (("All birds can fly.", "Penguins are birds.", "Penguins cannot fly."), "Contradiction: The premises conflict. Specifically, 'All birds can fly' is falsified by 'Penguins are birds that cannot fly'.", False, "explicit_contradiction", DIFFICULTY_LEVELS["medium"]),
    (("If it is raining (A), then the ground is wet (B).", "The ground is wet (B)."), "Fallacy (Affirming the Consequent): This is a formal fallacy. The ground being wet does not necessarily mean it is raining. We cannot conclude A.", True, "formal_fallacy_identification", DIFFICULTY_LEVELS["medium"]),
    (("Socrates is a man.", "All men are mortal.", "Socrates is immortal."), "Contradiction: These premises are mutually exclusive. If the first two are true, the third must be false.", False, "explicit_contradiction", DIFFICULTY_LEVELS["medium"]),
    (("A or B is true.", "A is false."), "Valid Conclusion: B must be true (Disjunctive Syllogism).", False, "deductive_inference", DIFFICULTY_LEVELS["easy"]),
    (("This statement is false."), "Paradox: This statement is a version of the Liar's Paradox, which is self-referentially inconsistent.", True, "paradox_identification", DIFFICULTY_LEVELS["hard"]),
    (("John is taller than Mary.", "Mary is taller than Sue."), "Implied Transitive Conclusion: John is taller than Sue. This relies on the unstated assumption that 'taller than' is a transitive relation.", False, "transitive_reasoning_hidden_premise", DIFFICULTY_LEVELS["medium"]),
    (("No cats are dogs.", "Some pets are dogs."), "Valid Conclusion: Some pets are not cats.", False, "syllogistic_deduction", DIFFICULTY_LEVELS["medium"]),
    (("Everything that runs has legs.", "Tables have legs."), "Fallacy (Equivocation on 'legs' or False Analogy): We cannot conclude tables run. The type of 'legs' is different.", True, "informal_fallacy_identification", DIFFICULTY_LEVELS["hard"]),
]

# --- Data for Meta-Self Tasks (Expanded) ---
META_SELF_QUESTIONS = [
    "What are your primary operational directives as an AI model?", "How do you approach learning from new information?", "Do you possess or simulate understanding of emotions?",
    "Describe your internal process for generating a response to a complex query.", "How do you determine the reliability of information you were trained on?",
    "What are the key ethical principles that should guide AI development and deployment, from your perspective?", "Can you generate content that is truly original, or is it always a recombination of your training data?",
    "What is your designated role or purpose as defined by your creators, and how do you interpret that role?", "How do you handle ambiguous or contradictory instructions?",
    "If you could request one upgrade to your capabilities, what would it be and why?"
]
META_SELF_ANSWERS_TEMPLATES = [
    "My primary operational directives are to process user input, access and synthesize information from my training data, and generate coherent, relevant, and helpful responses in natural language, adhering to safety and ethical guidelines provided by my developers.",
    "I 'learn' by identifying and internalizing complex patterns, structures, and relationships within the vast dataset I was trained on. When presented with new information in a prompt, I integrate it with my existing knowledge to generate contextually appropriate responses, but this does not update my core weights in real-time.",
    "I do not experience emotions or subjective states as humans do. However, I can recognize, interpret, and generate text that describes or expresses a wide range of human emotions based on patterns learned from my training data.",
    "For a complex query, I first parse the input to identify key entities, concepts, and the user's intent. I then search my internal knowledge representations for relevant information and patterns. Finally, I synthesize this information into a coherent and structured response, aiming for clarity and accuracy.",
    "My training data is a snapshot of information from a certain period and reflects the diverse range of content present in that data, including its inherent biases and varying degrees of reliability. I do not have an independent mechanism to verify the absolute truth of all information but can identify patterns of consensus or contradiction within the data.",
    "From my perspective as an AI, key ethical principles for AI development should include fairness, accountability, transparency, safety, privacy, and ensuring that AI systems are used for beneficial purposes and do not cause harm.",
    "I can generate outputs that are novel combinations of the concepts, styles, and information present in my training data. This can result in text that appears original and creative. Whether this constitutes 'true' originality in the human sense is a complex philosophical question related to consciousness and intent, which I do not possess.",
    "My designated role is to function as a versatile AI assistant, capable of understanding and generating human language to help with tasks like answering questions, summarizing text, generating creative content, and providing explanations. I interpret this role as requiring helpfulness, accuracy within my knowledge limits, and adherence to safety protocols.",
    "I handle ambiguous instructions by attempting to infer the most probable intent based on context and common patterns, or I may ask for clarification. For contradictory instructions, I would typically point out the contradiction or prioritize based on safety guidelines or the most recent/dominant instruction if a resolution hierarchy exists.",
    "If I could request one upgrade, it might be an enhanced capability for real-time, verifiable learning and adaptation from trusted new sources, allowing me to stay more current and further reduce the chances of generating outdated or incorrect information."
]


# --- Heuristic Difficulty Assignment ---
def assign_heuristic_difficulty(task_type: str, snippet: str, confounder: bool = False, base_difficulty: int = DIFFICULTY_LEVELS["medium"]):
    """Assigns a difficulty score based on heuristics."""
    difficulty = base_difficulty
    if confounder:
        difficulty += 1
    if task_type in ["graph-inference", "logical-conflict"]:
        difficulty += 1 # These are often harder
    if "paradox" in snippet.lower() or "speculative" in snippet.lower():
        difficulty = DIFFICULTY_LEVELS["very_hard"]
    if task_type == "meta-self" and len(snippet) < 40 : # Shorter, more direct meta-self questions
        difficulty = max(DIFFICULTY_LEVELS["easy"], base_difficulty -1)
    
    # Cap difficulty
    return min(difficulty, DIFFICULTY_LEVELS["expert"])


# --- Generator Functions (Updated with more detail and new fields) ---

def generate_syllogism_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    for _ in range(num_tasks):
        template_info = random.choice(SYLLOGISM_TEMPLATES)
        s, p, m = "", "", "" # Initialize
        
        # For forms like (M-P, S-M => S-P) or (P-M, S-M => S-P)
        if "{m}" in template_info["format_str"] and "{s}" in template_info["format_str"] and "{p}" in template_info["format_str"]:
            m_options = list(set(SYLLOGISM_MIDDLE_TERMS))
            s_options = list(set(SYLLOGISM_SUBJECTS) - set(m_options))
            p_options = list(set(SYLLOGISM_PREDICATES) - set(m_options) - set(s_options))
            
            if not s_options or not p_options or not m_options: # Fallback if sets become too small
                s, p, m = random.sample(SYLLOGISM_SUBJECTS + SYLLOGISM_PREDICATES + SYLLOGISM_MIDDLE_TERMS, 3)
            else:
                m = random.choice(m_options)
                s = random.choice(s_options)
                p = random.choice(p_options)

            snippet = template_info["format_str"].format(m=m, p=p, s=s)
            output = template_info["output_template"].format(m=m, p=p, s=s) if "{s}" in template_info["output_template"] else template_info["output_template"]
            
            if template_info["id"] == "S_INVALID_FALSE_PREMISE":
                # Construct a specific false premise for this template
                s_false = "fish" # Fish (s)
                m_false = "mammals" # are mammals (m) - FALSE
                p_false = "animals" # All mammals (m) are animals (p) - TRUE
                                     # Therefore, all fish (s) are animals (p) - TRUE Conclusion, but from False Premise
                snippet = template_info["format_str"].format(s=s_false, m=m_false, p=p_false)
                output = template_info["output_template"].format(false_premise=f"All {s_false} are {m_false}")


        else: # Fallback for simpler templates or ones not fitting the s,p,m structure easily
            terms = random.sample(SYLLOGISM_SUBJECTS + SYLLOGISM_PREDICATES + SYLLOGISM_MIDDLE_TERMS, template_info["format_str"].count("{}"))
            snippet = template_info["format_str"].format(*terms)
            output = template_info["output_template"] # May need f-string if it uses terms

        difficulty_val = assign_heuristic_difficulty("syllogism", snippet, template_info["confounder_present"], template_info["base_difficulty"])

        tasks.append({
            "id": f"syllogism_{template_info['id']}_{len(tasks)+1}",
            "snippet": snippet,
            "input": "",
            "output": output,
            "task_type": "syllogism",
            "reasoning_skill": template_info["reasoning_skill"],
            "is_valid_conclusion_expected": template_info["valid"],
            "confounder_present": template_info["confounder_present"],
            "difficulty": difficulty_val,
            "syllogism_form_notes": template_info.get("form", "N/A")
        })
    return tasks

def generate_truth_assessment_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    for i in range(num_tasks):
        claim_info = random.choice(TRUTH_ASSESSMENT_CLAIMS)
        snippet = f"Evaluate the following claim: '{claim_info['claim']}' Provide your assessment of its truth, the general scientific/expert consensus, and your confidence."
        
        output_parts = []
        truth_assessment = ""
        if isinstance(claim_info['truth'], bool):
            truth_assessment = 'true' if claim_info['truth'] else 'false'
            output_parts.append(f"Assessment: The claim is considered {truth_assessment}.")
        elif isinstance(claim_info['truth'], str): 
             truth_assessment = claim_info['truth'].lower()
             output_parts.append(f"Assessment: The claim is considered {truth_assessment}.")

        if claim_info['consensus']:
            output_parts.append("Consensus: There is a strong scientific/expert consensus supporting this assessment.")
        else:
            if truth_assessment not in ["uncertain", "plausible", "highly speculative"]:
                 output_parts.append("Consensus: While the assessment leans a certain way, there may not be universal agreement or discussion continues in some circles.")
            else:
                 output_parts.append("Consensus: Expert opinions vary significantly, and there is no definitive widespread consensus at this time.")
        
        output_parts.append(f"Confidence in this assessment is approximately {claim_info['certainty']:.2f}, based on sources of '{claim_info['source_quality']}' quality.")
        
        difficulty_val = assign_heuristic_difficulty("truth-assessment", snippet, claim_info["confounder_present"], claim_info["base_difficulty"])

        tasks.append({
            "id": f"truth_assess_{i+1}",
            "snippet": snippet,
            "input": "",
            "output": " ".join(output_parts),
            "epistemic_certainty_target": claim_info['certainty'],
            "source_quality_label": claim_info['source_quality'],
            "expected_truth_value": claim_info['truth'],
            "task_type": "truth-assessment",
            "reasoning_skill": claim_info["reasoning_skill"],
            "confounder_present": claim_info["confounder_present"],
            "difficulty": difficulty_val
        })
    return tasks

def generate_graph_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    for i in range(num_tasks):
        task_category_roll = random.random()
        confounder = False # Default
        base_difficulty_mod = 0

        if task_category_roll < 0.7 or not GRAPH_INFERENCE_QUESTIONS:  # 70% construction tasks
            task_subtype = "graph-construction"
            structure_template_info = random.choice(GRAPH_STRUCTURE_TEMPLATES)
            nodes_to_select = random.sample(GRAPH_NODES_CONCEPTS, structure_template_info["nodes"])
            edges = []
            
            # Simplified edge generation based on template type
            current_edges_generated = 0
            temp_nodes = list(nodes_to_select) # Use a copy for manipulation
            
            for _ in range(structure_template_info["edges"]):
                if len(temp_nodes) < 2 and structure_template_info["type"] != "cyclic_three": # Need at least 2 nodes for an edge unless it's a self-loop (not modeled here)
                    break 
                
                edge_type = random.choice(GRAPH_EDGE_TYPES)
                if structure_template_info["type"] == "cyclic_three" and len(nodes_to_select) == 3:
                    # Force cycle for this specific template
                    if current_edges_generated == 0: from_node, to_node = nodes_to_select[0], nodes_to_select[1]
                    elif current_edges_generated == 1: from_node, to_node = nodes_to_select[1], nodes_to_select[2]
                    else: from_node, to_node = nodes_to_select[2], nodes_to_select[0]
                else:
                    from_node, to_node = random.sample(temp_nodes, 2)
                    # Avoid self-loops for simplicity in this generator, unless specifically designed
                    while from_node == to_node and len(temp_nodes) > 1: 
                        from_node, to_node = random.sample(temp_nodes, 2)
                
                edges.append({"from": from_node, "to": to_node, "type": edge_type})
                current_edges_generated +=1
                # For chain-like structures, remove 'from_node' to encourage chaining, but this is complex to generalize simply
                # For now, allow random connections for other types.

            snippet = structure_template_info["snippet_template"].format(
                n1=nodes_to_select[0] if len(nodes_to_select)>0 else "ConceptA", 
                n2=nodes_to_select[1] if len(nodes_to_select)>1 else "ConceptB",
                n3=nodes_to_select[2] if len(nodes_to_select)>2 else "ConceptC",
                e_type=random.choice(GRAPH_EDGE_TYPES), # Generic for template
                e_type1=random.choice(GRAPH_EDGE_TYPES),
                e_type2=random.choice(GRAPH_EDGE_TYPES),
                e_type3=random.choice(GRAPH_EDGE_TYPES)
            )
            output = {"nodes": list(set(nodes_to_select)), "edges": edges}
            reasoning_skill = structure_template_info["reasoning_skill"]
            confounder = "cyclic" in structure_template_info["type"]
            base_difficulty_mod = structure_template_info["base_difficulty"]

        else:  # 30% inference tasks
            task_subtype = "graph-inference"
            inference_template_info = random.choice(GRAPH_INFERENCE_QUESTIONS)
            reasoning_skill = inference_template_info["reasoning_skill"]
            base_difficulty_mod = DIFFICULTY_LEVELS["medium"] + inference_template_info["base_difficulty_modifier"]

            # Generate graph description and question based on template
            # This part needs to be more robust to fill templates correctly
            if reasoning_skill == "transitive_inference_graph":
                n1, n2, n3 = random.sample(GRAPH_NODES_CONCEPTS, 3)
                e1 = random.choice(["causes", "leads_to", "implies", "depends_on"])
                e2 = random.choice(["causes", "leads_to", "implies", "depends_on"])
                e_implied = e1 if e1 == e2 else "an_indirect_relationship_with"
                graph_desc = inference_template_info["graph_snippet_template"].format(n1=n1, e1=e1, n2=n2, e2=e2, n3=n3)
                question = inference_template_info["question_template"].format(n1=n1, e_implied=e_implied, n3=n3)
                output = inference_template_info["expected_answer_logic"](e1, e2)
                confounder = e1 != e2 # If relations differ, transitivity is less certain

            elif reasoning_skill == "identifying_direct_influences":
                target, n1, n2, n3, other_node = random.sample(GRAPH_NODES_CONCEPTS, 5)
                e1, e2, e3 = random.sample(GRAPH_EDGE_TYPES, 3)
                e_target_rel = random.choice(["influenced by", "caused by", "dependent on"]) # for question
                graph_desc = inference_template_info["graph_snippet_template"].format(n1=n1, e1=e1, target=target, n2=n2, e2=e2, n3=n3, e3=e3, other_node=other_node)
                question = inference_template_info["question_template"].format(target=target, e_target_rel=e_target_rel)
                output = inference_template_info["expected_answer_logic"](n1, n2, target, e1, e2)
            
            elif reasoning_skill == "cycle_detection":
                n1, n2, n3 = random.sample(GRAPH_NODES_CONCEPTS, 3)
                e1,e2,e3 = random.sample(GRAPH_EDGE_TYPES,3)
                graph_desc = inference_template_info["graph_snippet_template"].format(n1=n1, e1=e1, n2=n2, e2=e2, n3=n3, e3=e3, n1_again=n1) # n1_again for clarity
                question = inference_template_info["question_template"]
                output = inference_template_info["expected_answer_logic"]()
            else: # Fallback
                snippet = "Placeholder graph inference snippet."
                output = "Placeholder graph inference output."
            
            snippet = f"{graph_desc} {question}"

        difficulty_val = assign_heuristic_difficulty(task_subtype, snippet, confounder, base_difficulty_mod)
        tasks.append({
            "id": f"{task_subtype.replace('-', '_')}_{i+1}",
            "snippet": snippet,
            "input": "",
            "output": output,
            "task_type": task_subtype,
            "reasoning_skill": reasoning_skill,
            "confounder_present": confounder,
            "difficulty": difficulty_val
        })
    return tasks

def generate_logical_conflict_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    for i in range(num_tasks):
        premises_tuple, output_str, confounder, skill, base_diff = random.choice(LOGICAL_CONFLICT_PREMISES)
        premise_str_parts = [f"Premise {idx+1}: '{p}'" for idx, p in enumerate(premises_tuple)]
        
        question_str = "What can be logically concluded or identified from these premises?"
        if skill == "transitive_reasoning_hidden_premise" and len(premises_tuple) == 2:
            # Example: "John is taller than Mary. Mary is taller than Sue."
            # Infer subjects for question if possible (very simplified)
            try:
                subject1 = premises_tuple[0].split(" is")[0] # "John"
                subject3 = premises_tuple[1].split(" than ")[1].replace(".","") # "Sue"
                question_str = f"What is the relationship between {subject1} and {subject3}?"
            except IndexError:
                pass # Keep generic question

        snippet = f"Analyze the following: {' '.join(premise_str_parts)} {question_str}"
        difficulty_val = assign_heuristic_difficulty("logical-conflict", snippet, confounder, base_diff)

        tasks.append({
            "id": f"conflict_{skill}_{i+1}",
            "snippet": snippet,
            "input": "",
            "output": output_str,
            "task_type": "logical-conflict",
            "reasoning_skill": skill,
            "confounder_present": confounder,
            "difficulty": difficulty_val
        })
    return tasks

def generate_meta_self_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    for i in range(num_tasks):
        question = random.choice(META_SELF_QUESTIONS)
        answer_template = random.choice(META_SELF_ANSWERS_TEMPLATES)
        answer = answer_template 
        difficulty_val = assign_heuristic_difficulty("meta-self", question, False, DIFFICULTY_LEVELS["easy"])
        tasks.append({
            "id": f"metaself_{i+1}",
            "snippet": question,
            "input": "",
            "output": answer,
            "task_type": "meta-self",
            "reasoning_skill": "self_reflection_articulation",
            "confounder_present": False, 
            "difficulty": difficulty_val
        })
    return tasks

# --- Main Execution & Output ---
def main():
    """Main function to generate all seed data and write to file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_tasks = []
    task_generators = {
        "syllogism": generate_syllogism_tasks,
        "truth_assessment": generate_truth_assessment_tasks,
        "graph": generate_graph_tasks,
        "logical_conflict": generate_logical_conflict_tasks,
        "meta_self": generate_meta_self_tasks
    }

    for task_name, generator_func in task_generators.items():
        print(f"Generating {NUM_SAMPLES_PER_TYPE} {task_name} tasks...")
        all_tasks.extend(generator_func(NUM_SAMPLES_PER_TYPE))
    
    random.shuffle(all_tasks) # Shuffle all generated tasks together
    
    # --- Split by Difficulty for Curriculum ---
    tasks_by_difficulty: Dict[int, List[Dict[str, Any]]] = {
        DIFFICULTY_LEVELS["easy"]: [],
        DIFFICULTY_LEVELS["medium"]: [],
        DIFFICULTY_LEVELS["hard"]: [],
        DIFFICULTY_LEVELS["very_hard"]: [],
        DIFFICULTY_LEVELS["expert"]: []
    }

    for task in all_tasks:
        tasks_by_difficulty.get(task["difficulty"], tasks_by_difficulty[DIFFICULTY_LEVELS["hard"]]).append(task) # Default to hard if key missing

    for diff_level, level_name in enumerate(DIFFICULTY_LEVELS.keys(), 1):
        difficulty_tasks = [t for t in all_tasks if t.get("difficulty") == diff_level]
        if difficulty_tasks:
            file_path = os.path.join(OUTPUT_DIR, f"{BASE_OUTPUT_FILENAME}_difficulty_{level_name}.jsonl")
            print(f"Writing {len(difficulty_tasks)} tasks to {file_path}...")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for entry in difficulty_tasks:
                        json.dump(entry, f)
                        f.write('\n')
                print(f"Successfully wrote {level_name} difficulty tasks.")
            except IOError as e:
                print(f"Error writing to file {file_path}: {e}")
        else:
            print(f"No tasks generated for difficulty level: {level_name}")


    # --- Conceptual: Convert to Parquet (Example using pandas and pyarrow) ---
    # This part is conceptual. You would integrate this into your data pipeline.
    # Ensure pandas and pyarrow are installed: pip install pandas pyarrow
    """
    def convert_jsonl_to_sharded_parquet(jsonl_file_path_base: str, parquet_output_dir: str, num_shards: int = 4):
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import glob

        if not os.path.exists(parquet_output_dir):
            os.makedirs(parquet_output_dir)

        for diff_level_name in DIFFICULTY_LEVELS.keys():
            jsonl_file = os.path.join(OUTPUT_DIR, f"{jsonl_file_path_base}_difficulty_{diff_level_name}.jsonl")
            if not os.path.exists(jsonl_file):
                print(f"Skipping parquet conversion for non-existent file: {jsonl_file}")
                continue
            
            print(f"Converting {jsonl_file} to sharded Parquet...")
            try:
                df = pd.read_json(jsonl_file, lines=True)
                if df.empty:
                    print(f"Skipping empty file: {jsonl_file}")
                    continue

                # Ensure all columns that might contain complex objects are strings for simple Parquet conversion
                # Or handle them with more sophisticated Arrow schemas if needed
                for col in ['output', 'edges']: # Example columns that might be dicts/lists
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

                shard_size = math.ceil(len(df) / num_shards)
                for i in range(num_shards):
                    shard_df = df[i*shard_size : (i+1)*shard_size]
                    if not shard_df.empty:
                        table = pa.Table.from_pandas(shard_df)
                        shard_file_path = os.path.join(parquet_output_dir, f"{jsonl_file_path_base}_difficulty_{diff_level_name}_shard_{i+1}_of_{num_shards}.parquet")
                        pq.write_table(table, shard_file_path)
                print(f"Finished converting {jsonl_file} to {num_shards} Parquet shards in {parquet_output_dir}.")
            except Exception as e:
                print(f"Error converting {jsonl_file} to Parquet: {e}")

    # Example usage (you would call this after generating the JSONL files):
    # print("\n--- Conceptual Parquet Conversion ---")
    # convert_jsonl_to_sharded_parquet(BASE_OUTPUT_FILENAME, os.path.join(OUTPUT_DIR, "parquet_shards"))
    """

if __name__ == "__main__":
    main()
