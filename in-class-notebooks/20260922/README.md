# September 22 materials

This lesson uses the traveling salesman problem (TSP) to practice directing an
AI agent through planning, implementation, review, and revision. The exercise
recreates the app demonstrated in class from the earlier TSP notebook.

The September 22 lecture notes are distributed through Blackboard. They explain
the vocabulary and classroom discussion. This guide provides the files and
instructions for the demonstrations.

## Materials

- The [slides](agentic-ai.html) are the draft deck used in class. They also contain
  supplemental material that the lecture did not cover in detail.
- The [spoken build prompt](transcript.txt) is the unchanged seven-minute
  dictation used to request the app. Its repetition and transcription errors are
  intentional evidence of how the request began. It is not the class transcript
  or a complete log of the agent's work.
- The [starting project](../TSP/) is the notebook used in the preceding classes.
- The [finished app](TSP-demo/README.md) provides a comparison after the exercise.
- The [review report](TSP-demo/process_report.html) describes the original build
  and subsequent reviews, with screenshots and recorded checks.
- The [skill creation prompt](ai-update-prompt.md) recreates the daily AI updates
  demonstration through a separate conversation.

## Download and open the files

On the [repository home page](../../), choose **Code**, then **Download ZIP**,
and extract the download. Open `in-class-notebooks/20260922` in the extracted
folder. An existing Git clone can instead be updated with `git pull` after
preserving local work.

Open `agentic-ai.html` in a browser to view the slides. Open
`TSP-demo/process_report.html` to read the illustrated report. Keep the report
beside its `report_images` folder so the images load. The HTML files are local
browser documents; the links on GitHub display their source. The Streamlit app
requires the separate launch command below.

## Recreate the TSP build

### Prepare a working copy

1. Copy the entire `in-class-notebooks/TSP` folder to a separate location, such
   as a folder named `TSP-practice` on the Desktop. Start from `TSP`, not the
   finished `20260922/TSP-demo` folder.
2. Copy this lesson's `transcript.txt` into `TSP-practice`, beside `pixi.toml`
   and `tsp_demo.py`. Keep the filename `transcript.txt`.
3. Open a terminal in `TSP-practice`. The commands below must run in that
   directory. If changing directories with `cd`, put paths containing spaces
   in quotes.
4. Install and check the starting project:

```bash
pixi install
pixi run check
pixi run test
pixi run demo
```

The last command opens the Marimo notebook. Stop its terminal process with
Ctrl+C when finished. The starting notebook constructs a nearest-neighbor tour
from each starting facility, selects the shortest, and improves it with sampled
subsequence reversals. The project README explains the data and road-distance
assumptions.

### Start the agent

Install and sign in to Codex using the
[official CLI instructions](https://learn.chatgpt.com/docs/codex/cli), then run
the following in `TSP-practice`:

```bash
codex
```

In Codex, enter `/plan`, then send:

```text
Read this transcript and proceed.
```

Answer the clarification questions and check that the proposed plan describes
the intended app before selecting implementation. In a desktop interface,
select the same working folder and use its planning controls. The instructor's
`codex-dsp` alias is specific to his machine. Plain `codex` is the command for
this exercise. See the official
[command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
for `/plan`, `/model`, and `/status`.

The raw prompt names models from the instructor's setup. Ask the agent to use
models and review tools available in the local account. If separate subagents
are unavailable, request distinct sequential review passes and record that
difference. The point is to inspect findings and revisions, not to reproduce
the same model roster.

### Review the result

The requested app should select states, construct and improve a route, display
progress, and show the resulting tour. Ask the agent for the exact launch and
test commands for its implementation. Keep the original data files unchanged.

Use the following checks while reviewing the build:

- Confirm that every selected facility appears exactly once before the route
  returns to its start, and that the reported distance includes that return.
- Check that the improved route is no longer than the starting route. Try a
  different state selection and check that the displayed result matches it.
- Ask for measured solve time and the settings used. Compare runs with the same
  inputs, seed, and stopping rule.
- Require findings and proposed repairs alongside review scores. The main agent
  should verify a finding before implementing a change.
- Open the app and inspect the actual map, text, controls, and progress display.
  Try a narrower browser window as well as the desktop view.

The instructor's first review did not inspect a rendered browser page. A later
browser review found layout and resizing problems. After the initial build,
the following request makes that additional check explicit:

```text
Open the app in a browser and test its controls. Take screenshots of the
selection view and a completed route at desktop and narrow widths. Inspect
the screenshots for overlapping text, unreadable labels, wasted space, and
map resizing problems. Report concrete findings and recommended fixes.
Verify each finding, apply the useful fixes, and inspect the app again.
Use at most three review rounds. If browser access is unavailable, say so
and tell me which views I need to inspect myself.
```

This is a practice exercise, with no separate submission specified here.
Use the same pattern when exploring the course project briefs and data.

## Compare with the finished example

From the extracted repository root, run:

```bash
cd in-class-notebooks/20260922/TSP-demo
pixi install
pixi run app
```

Open the local address printed in the terminal. Select states and choose
**Find a route**. Stop the server with Ctrl+C. Run `pixi run test` and
`pixi run check` from this same folder to check the finished example.

The app plots straight segments between facilities while calculating distance
from the road-distance table. Its search finds a heuristic route and does not
prove optimality. Different generated implementations can produce different
routes, layouts, and runtimes.

Read the report after trying the build. Its tests and times describe the
instructor's original run. Its review scores are judgments by the reviewing
agents, and scores from the code review and browser review used different
evidence. A rising score alone does not establish that an app works.

## Project preparation

Form teams of up to three, inspect the seven project briefs and data bundles on
Blackboard, and email the chosen project to Dr. Freeman by **October 1, 2026**.
Blackboard remains the authoritative source for project requirements, grading,
and deadlines. The next class returns to scheduling.
