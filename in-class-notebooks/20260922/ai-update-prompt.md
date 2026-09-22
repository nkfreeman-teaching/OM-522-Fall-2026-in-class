# Create a daily AI updates skill

The class demonstration began with a request for a reusable daily briefing.
The agent asked questions about coverage, ranking, the first search window,
and delivery before creating the skill. This prompt combines the initial
request with the choices made in class. It is an edited reconstruction, not
a verbatim development transcript or a finished skill.

Create a separate folder named `ai-update`, open it in Codex, enter plan mode,
and send the following prompt:

```text
Help me build a skill that I can invoke to get a daily briefing of up to
five consequential developments in AI worldwide.

Cover research, model releases, products and developer tools, policy, and
significant community developments. Rank by broad impact and strength of
evidence. Do not force one item from each category.

Search since the last successful briefing. On the first run, cover the
preceding seven days. Remember which stories were covered so later runs
do not repeat them unless something material has changed.

Give short analysis in chat with dates and direct source links. Explain
what happened and why it matters. Prefer original sources and distinguish
reported claims from established findings. Include fewer than five items
when there are not five consequential, well-supported developments.

Ask any questions needed to understand the request, then propose a plan.
After we agree on it, create the skill using the skill creator available
in this installation. Show where the skill and its run history will live,
and explain exactly how to invoke it. Keep the briefing history separate
from the reusable skill instructions.
```

Review the plan, resolve the remaining questions, and select implementation.
Then inspect the generated `SKILL.md` and ask the agent to explain any unclear
instruction. Invoke the skill once and check the source dates and links in the
briefing. A request for a daily briefing defines a reusable task; automatic
scheduling would require a separate setup.

Codex CLI supports explicit skill selection by typing `$` and selecting a
skill, or by using `/skills`. The generated skill's name determines the exact
invocation. The official [skill guide](https://learn.chatgpt.com/docs/build-skills)
explains `SKILL.md`, the built-in creator, and skill discovery.
