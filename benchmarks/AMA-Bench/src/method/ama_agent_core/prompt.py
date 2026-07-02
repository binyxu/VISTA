CODE_GENERATION_PROMPT_TEMPLATE = """You are helping to extract relevant information from a trajectory to answer a question by writing Python code.

**Question:** {query}

**Task:** {task}

**Trajectory Format Reference (first 2 turns):**
{trajectory_sample}

**Trajectory Data (JSON format):**
Available as variable `trajectory_json` with structure:
{{
  "trajectory": [
    {{
      "turn_idx": 0,      // Turn number (int)
      "action": "...",    // Action taken at this turn (string)
      "observation": "..." // Environment observation after action (string)
    }},
    ...
  ],
  "task": "...",        // Task description (string)
  "episode_id": "..."   // Episode identifier (string)
}}

**Your Task:**
Write Python code that processes the trajectory JSON and extracts the relevant information to answer the question.

**Available Examples:**

Example 1: Finding specific actions
```python
import json

# Parse trajectory
trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find turns where a specific action was taken
relevant_turns = []
for turn in trajectory:
    if 'pick up' in turn.get('action', '').lower():
        relevant_turns.append({{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }})

result = {{
    'relevant_turns': relevant_turns,
    'count': len(relevant_turns)
}}
```



Example 2: Finding when something happened (until turn X)
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find first turn when door was opened
first_open = None
for turn in trajectory:
    if 'open door' in turn.get('action', '').lower():
        first_open = turn['turn_idx']
        break

# Get all turns until that point
turns_until_open = [t for t in trajectory if t['turn_idx'] <= first_open] if first_open else []

result = {{
    'event': 'door opened',
    'first_occurrence': first_open,
    'turns_until_event': len(turns_until_open)
}}
```

Example 3: Finding last occurrence of something
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find last turn where agent picked up something
last_pickup = None
for turn in reversed(trajectory):
    if 'pick up' in turn.get('action', '').lower():
        last_pickup = {{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }}
        break

result = {{'last_pickup': last_pickup}}
```

Example 4: Causal relationship - what happened after X
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find when key was picked up, then what happened next
key_pickup_turn = None
for turn in trajectory:
    if 'key' in turn.get('action', '').lower() and 'pick' in turn.get('action', '').lower():
        key_pickup_turn = turn['turn_idx']
        break

# Get next 3 turns after picking up key
subsequent_actions = []
if key_pickup_turn is not None:
    for turn in trajectory:
        if turn['turn_idx'] > key_pickup_turn and len(subsequent_actions) < 3:
            subsequent_actions.append({{
                'turn': turn['turn_idx'],
                'action': turn['action']
            }})

result = {{
    'trigger_event': 'picked up key',
    'trigger_turn': key_pickup_turn,
    'subsequent_actions': subsequent_actions
}}
```

**Instructions:**
1. Write Python code that processes the trajectory JSON (available as variable `trajectory_json`)
2. Extract information relevant to answering the question
3. The code should be self-contained and executable
4. Store the final result in a variable named `result`

**Robust matching guidelines (counting / search / aggregation):**
- Use case-insensitive matching (`.lower()`) and search BOTH `action` and `observation` fields unless the question restricts to one. The same target may appear with different quoting/brackets — match a substring or use a regex.
- Always return BOTH the count AND the list of matching turn indices (with a short evidence snippet) in `result`, so the answering LLM can verify and use exact turn ids.
- Do not double-count the same turn for the same event.

**Output Format:**
You MUST format your response as follows:

**CODE**:
```python
# Your Python code here
```

Important: The code must be wrapped with **CODE**: marker followed by ```python code block.
<think><\think>
"""

COMPRESS_PROMPT_TEMPLATE = """You are presented with a section of agent trajectory (actions and observations). Compress it into a state memory that future readers can use to answer detailed questions about what happened.

Task: {task}

Trajectory Section:
{trajectory_text}

{previous_state_text}

Identify KEY turns — turns where state meaningfully changed, a command/query/code was executed, an important result/value/schema/error appeared, or the agent shifted sub-goal. For each key turn record an env_state block with bullet keys that BEST FIT this turn. Common keys include: action, command, query, location, inventory, schema, sql, result, finding, error, change. Use whatever keys fit; do NOT force fields that do not apply, and do NOT dump unrelated context into a "change" field.

CRITICAL — copy these VERBATIM from the trajectory; never paraphrase or summarize:
- commands, queries, code (SQL, shell, Python, function/tool calls)
- identifiers: file paths, URLs, table/column/schema names, IDs, UI element ids and selectors, entity names
- numeric values, counts, computed results, dates, response codes
- error messages

Also write a one-line Memory Summary describing the section's overall progress.

Output everything after the marker below.

**STATE_MEMORY**

memory_summary: key_turns=<turn ids>; key_objective=<main objective>; key_events=<critical events, commands, state changes, key results>

turn_id: <turn number>
env_state:
- <key>: <value>
- ...


turn_id: <turn number>
env_state:
- <key>: <value>


Constraints:
(1) NEVER paraphrase identifiers, commands, queries, paths, URLs, numbers, or error messages — copy them verbatim.
(2) Only record turns matching the KEY-turn criteria above.
(3) Each env_state bullet should describe ONLY what is salient at that turn — do not repeat unchanged context from earlier turns and do not include unrelated environment dumps.
(4) Use consistent entity / object names across all turns.
(5) When previous state memory is provided, retain its details and integrate new findings.
(6) memory_summary must include key_turns, key_objective, and key_events on a single line.
(7) Do not invent facts. Do NOT write SQL queries, code, or commands that the agent did not actually execute in the trajectory — only record what the agent actually did. Do NOT add an "Explanation", "Analysis", or "Recommended Approach" section.
(8) Output ONLY the state memory in the prescribed format after **STATE_MEMORY**. No prose, no markdown sections beyond memory_summary/turn_id/env_state.
"""

CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE = """You are routing a question about an agent trajectory to the right retrieval stage.

Query: {query}

Retrieved Turns (top-k from similarity search — this is NOT the full trajectory):
{retrieved_chunks}

IMPORTANT: the retrieved turns above are a small subset of the full trajectory selected by similarity. Counts, lists, and aggregations computed over this subset WILL UNDERCOUNT and MUST NOT be used to answer "how many" / "count" / "list all" type questions.

# Decision procedure — follow in order, stop at the first match

Step 1. Does the question ask any of the following?
  - "how many ...", "count of ...", "total number of ...", "frequency of ..."
  - "how often", "how many times did X happen"
  - "list all X", "find every X", "all distinct/unique X"
  - sum / average / percentage / aggregation
  - patterns spanning many turns ("redundant loop", "repeated actions", "every time X")
  - a tally or breakdown of tool/command/action types across turns
  → If YES, you MUST choose NEED_CODE. Do NOT answer from the retrieved subset even if it appears to contain examples — you will undercount. End here.

Step 2. Otherwise, can you answer the question COMPLETELY and ACCURATELY using only the retrieved turns above (plus general knowledge)?
  - "Completely": every part of the question is addressed.
  - "Accurately": you can point to specific turn(s) in the retrieved set that justify each part of the answer.
  → If YES, choose SUFFICIENT and provide the full answer. End here.

Step 3. Are there specific other turns (adjacent, range, or by index) you need to inspect to answer? For example, the answer hinges on a turn that is referenced but not shown, or you need context around a retrieved turn.
  → If YES, choose NEED_GRAPH and specify which turns. End here.

Step 4. If you reach this step, the question requires reasoning over many parts of the trajectory that the similarity search did not surface.
  → Choose NEED_CODE.

# Output formats — use EXACTLY one of these, on its own line(s) at the start of your response

SUFFICIENT
ANSWER: <your complete and accurate answer>

NEED_GRAPH: <spec>
  where <spec> is one or more of:
    turn_5 before=2 after=1
    turn_8 before=3 after=0, turn_15 before=0 after=2
    turns 5 to 10
    turns 3 to 8, turns 15 to 20
    turns 3, 7, 12, 18

NEED_CODE: <short description of what to compute>

Response:
"""

TOOL_USE_PROMPT_TEMPLATE = """You are helping retrieve relevant information from a trajectory to answer a question.

**Question:** {query}

**Available Tools:**

You have access to TWO powerful tools to search and retrieve information from the trajectory:

1. **traj_find** - Locates relevant turns
   - Purpose: Search for specific keywords/entities/actions in the trajectory
   - Parameters:
     * query (required): The search term (e.g., "open door", "key", "red box")
     * mode (optional): Search strategy
       - "keyword": Search anywhere in text (default)
       - "action": Search only in action field
       - "entity": Search for specific entity mentions
   - Returns: List of turn indices where the query was found
   - Example: traj_find(query="pick up", mode="action")

2. **traj_get** - Retrieves detailed information
   - Purpose: Get full details from specific turns
   - Parameters:
     * span (required): Which turns to get
       - {{"indices": [1, 2, 3]}} for specific turns
       - {{"start": 1, "end": 5}} for a range
     * fields (optional): What info to include ["action", "observation", "action_space"]
   - Returns: Formatted text with detailed turn information
   - Example: traj_get(span={{"indices": [5, 7, 9]}})

**Recommended Strategy:**
1. Use traj_find to locate turns related to the question
2. Use traj_get to retrieve detailed information from those turns
3. You can call tools multiple times to gather complete information

**Your Task:**
Use these tools strategically to find and retrieve ALL relevant information needed to answer the question thoroughly."""

ANSWER_WITH_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory and retrieved detailed information, provide a natural language answer to the query.

Query: {query}

State Memory (compressed):
{state_mem_str}

Retrieved Detailed Information:
{relevant_mem}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

ANSWER_WITHOUT_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory, provide a natural language answer to the query.

Query: {query}

State Memory:
{state_mem_str}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

CAUSAL_PROMPT_TEMPLATE = """You are analyzing a trajectory to extract causal relationships between events and state changes.

Task: {task}

Trajectory:
{trajectory_text}

{previous_state_text}

Your task is to identify and extract causal relationships from the trajectory.

For each causal relationship, identify:
1. The CAUSE: an action or event that triggers a change
2. The EFFECT: the resulting state change or consequence
3. The TURN(S): when this causal relationship occurs

Output your response after the markers below.

**CAUSAL_GRAPH**
[
  {{
    "cause": "description of triggering action/event",
    "effect": "description of resulting state change",
    "cause_turn": <turn number>,
    "effect_turn": <turn number>,
    "entities": ["entity1", "entity2"]
  }},
  ...
]

**STATE_MEMORY**
[Your state memory content here]
"""
