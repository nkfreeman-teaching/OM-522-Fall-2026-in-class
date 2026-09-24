const byId = id => document.getElementById(id);
const metricNames = {
  completion: 'Total completion time',
  weighted_completion: 'Weighted total completion',
  lateness: 'Maximum lateness',
  weighted_tardiness: 'Weighted tardiness'
};
const familyNames = {backlog: 'Backlog', steady: 'Steady arrivals', rush: 'Rush orders'};
const ruleNames = {
  EDD: 'Earliest due date',
  SPT: 'Shortest processing time',
  CR: 'Critical ratio',
  WSPT: 'Weighted shortest processing time'
};
const moveNames = {adjacent: 'adjacent swap', swap: 'any-pair swap', reversal: 'subsequence reversal'};
const methodShort = method => {
  const [rule, move] = method.split('|');
  return rule + ' + ' + moveNames[move];
};
const methodLong = method => {
  const [rule, move] = method.split('|');
  return ruleNames[rule] + ' with ' + (move === 'reversal' ? 'subsequence reversals' : moveNames[move] + 's');
};
const formatPercent = value => value > 0 && value < 0.00005 ? '<0.01%' : (value * 100).toFixed(2) + '%';
const formatPoints = value => (value > 0 ? '+' : '') + (value * 100).toFixed(2) + ' pp';
const number = value => Math.round(value).toLocaleString();
const average = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const allCells = method => DATA.families.flatMap(family => DATA.metrics.map(metric => DATA.cells[method][family][metric]));

function score(method, priority) {
  const cells = DATA.cells[method];
  if (priority === 'robust') return Math.max(...allCells(method));
  if (priority === 'average') return average(allCells(method));
  const [kind, item] = priority.split(':');
  if (kind === 'metric') return average(DATA.families.map(family => cells[family][item]));
  return average(DATA.metrics.map(metric => cells[item][metric]));
}

function rankedMethods() {
  const priority = byId('priority').value;
  return [...DATA.methods].sort((first, second) =>
    score(first, priority) - score(second, priority) ||
    score(first, 'robust') - score(second, 'robust') ||
    first.localeCompare(second)
  );
}

function scoreName(priority) {
  if (priority === 'robust') return 'Largest group-average gap';
  if (priority === 'average') return 'Average of the 12 group gaps';
  const [kind, item] = priority.split(':');
  return kind === 'metric' ? 'Average gap for ' + metricNames[item].toLowerCase() :
    'Average gap for ' + familyNames[item].toLowerCase();
}

let inspected = '';
let compared = '';
let showAll = false;

function renderHero(ranked) {
  const priority = byId('priority').value;
  const first = ranked[0];
  const second = ranked[1];
  const tied = ranked.filter(method => Math.abs(score(method, priority) - score(first, priority)) < 1e-12);
  byId('lead-label').textContent = tied.length > 1 ? 'Leader after the group-gap tie-break' : 'Recommended method for the selected priority';
  byId('lead-method').textContent = methodLong(first);
  byId('lead-description').textContent = 'Choose the next available job by ' + ruleNames[first.split('|')[0]].toLowerCase() + ', then improve the order with ' + moveNames[first.split('|')[1]] + ' moves.';
  byId('lead-score').textContent = formatPercent(score(first, priority));
  byId('score-label').textContent = scoreName(priority);
  byId('lead-runner').textContent = tied.length > 1 ?
    tied.length + ' methods tie on this priority. This method has the lower default score among them, at ' + formatPercent(score(first, 'robust')) + '.' :
    'Next method: ' + methodShort(second) + ' at ' + formatPercent(score(second, priority)) + '.';
  const stability = byId('stability');
  stability.hidden = priority !== 'robust';
  if (priority === 'robust') {
    const frequency = DATA.uncertainty[first].winner_frequency;
    byId('stability-text').innerHTML = '<strong>' + methodShort(first) + '</strong> ranked first in <strong>' + number(frequency * 2000) + ' of 2,000</strong> resampled versions of these cases (' + (frequency * 100).toFixed(1) + '%).';
  }
}

function cellDifferences() {
  return DATA.families.flatMap(family => DATA.metrics.map(metric => ({
    family,
    metric,
    selected: DATA.cells[inspected][family][metric],
    other: DATA.cells[compared][family][metric],
    difference: DATA.cells[inspected][family][metric] - DATA.cells[compared][family][metric]
  })));
}

function renderFocus() {
  byId('focus-title').textContent = 'Inspecting ' + methodShort(inspected) + ' against ' + methodShort(compared);
  byId('compare').innerHTML = DATA.methods.filter(method => method !== inspected).map(method =>
    '<option value="' + method + '"' + (method === compared ? ' selected' : '') + '>' + methodShort(method) + '</option>'
  ).join('');
  const differences = cellDifferences();
  const strongestGain = differences.reduce((best, item) => item.difference < best.difference ? item : best);
  const strongestLoss = differences.reduce((worst, item) => item.difference > worst.difference ? item : worst);
  const summary = [];
  if (strongestGain.difference < -1e-10) {
    summary.push('Largest advantage: ' + Math.abs(strongestGain.difference * 100).toFixed(2) + ' percentage points lower for ' + metricNames[strongestGain.metric].toLowerCase() + ' with ' + familyNames[strongestGain.family].toLowerCase() + '.');
  }
  if (strongestLoss.difference > 1e-10) {
    summary.push('Largest trade-off: ' + (strongestLoss.difference * 100).toFixed(2) + ' percentage points higher for ' + metricNames[strongestLoss.metric].toLowerCase() + ' with ' + familyNames[strongestLoss.family].toLowerCase() + '.');
  }
  byId('comparison-summary').textContent = summary.join(' ') || 'These methods tie on all twelve group-average gaps.';
  let table = '<table class="matrix"><thead><tr><th>Arrival pattern</th>';
  table += DATA.metrics.map(metric => '<th>' + metricNames[metric] + '</th>').join('');
  table += '</tr></thead><tbody>';
  for (const family of DATA.families) {
    table += '<tr><th>' + familyNames[family] + '</th>';
    for (const metric of DATA.metrics) {
      const item = differences.find(value => value.family === family && value.metric === metric);
      const direction = item.difference < -1e-10 ? 'better' : item.difference > 1e-10 ? 'worse' : 'equal';
      table += '<td data-label="' + metricNames[metric] + '"><span class="delta ' + direction + '">' + formatPoints(item.difference) + '</span>';
      table += '<span class="pair-values">' + formatPercent(item.selected) + ' vs ' + formatPercent(item.other) + '</span></td>';
    }
    table += '</tr>';
  }
  byId('matrix').innerHTML = table + '</tbody></table>';
}

function renderRanking(ranked) {
  const priority = byId('priority').value;
  byId('ranking-title').textContent = 'Methods ranked by ' + scoreName(priority).toLowerCase();
  byId('ranking-description').textContent = 'Smaller is better. ' + (showAll ? 'All twelve methods appear below.' : 'The first four methods appear below.') + ' This ranking uses the measure each schedule was optimized to improve.';
  const displayed = showAll ? ranked : ranked.slice(0, 4);
  byId('ranking-list').innerHTML = displayed.map((method, index) =>
    '<button class="ranking-row" type="button" data-method="' + method + '" aria-pressed="' + (method === inspected) + '">' +
    '<span class="ordinal">' + (index + 1) + '.</span><span class="method">' + methodShort(method) + '</span>' +
    '<span class="value">' + formatPercent(score(method, priority)) + '</span></button>'
  ).join('');
  for (const row of byId('ranking-list').querySelectorAll('[data-method]')) {
    row.onclick = () => {
      inspected = row.dataset.method;
      if (inspected === compared) compared = ranked.find(method => method !== inspected);
      render();
    };
  }
  byId('show-all').textContent = showAll ? 'Show four leaders' : 'Show all 12 methods';
  byId('show-all').setAttribute('aria-expanded', String(showAll));
}

function crossGap(method, target, family, metric) {
  const rows = DATA.rows.filter(row => row.m === method && row.o === target && (family === 'all' || row.f === family));
  return average(rows.map(row => {
    const best = DATA.best_all[row.f + '|' + row.i + '|' + metric];
    return (row.v[metric] - best) / Math.max(1, Math.abs(best));
  }));
}

function renderTradeoffs() {
  const target = byId('target').value;
  const family = byId('family').value;
  byId('tradeoff-title').textContent = 'Trade-offs when optimizing ' + metricNames[target].toLowerCase();
  byId('tradeoff-pair').textContent = 'Inspected: ' + methodShort(inspected) + '. Compared: ' + methodShort(compared) + '.';
  let table = '<table><thead><tr><th>Outcome</th><th>Inspected</th><th>Compared</th><th>Difference</th></tr></thead><tbody>';
  for (const metric of DATA.metrics) {
    const first = crossGap(inspected, target, family, metric);
    const second = crossGap(compared, target, family, metric);
    const difference = first - second;
    table += '<tr' + (metric === target ? ' class="target-row"' : '') + '><td>' + metricNames[metric] + '</td>';
    table += '<td data-label="Inspected">' + formatPercent(first) + '</td><td data-label="Compared">' + formatPercent(second) + '</td>';
    table += '<td class="difference" data-label="Difference">' + formatPoints(difference) + '</td></tr>';
  }
  byId('cross-table').innerHTML = table + '</tbody></table><p class="small muted" style="margin: 9px 0 0">The shaded row is the search target. Difference means inspected minus compared, in percentage points. Negative is better.</p>';
  const relevant = method => DATA.rows.filter(row => row.m === method && row.o === target && (family === 'all' || row.f === family));
  const first = relevant(inspected);
  const second = relevant(compared);
  byId('effort').innerHTML = '<div><strong>' + number(average(first.map(row => row.e))) + '</strong><span>Candidate schedules scored per search, versus ' + number(average(second.map(row => row.e))) + ' for the comparison</span></div>' +
    '<div><strong>' + number(average(first.map(row => row.s)) * 1000) + ' ms</strong><span>Mean search time, versus ' + number(average(second.map(row => row.s)) * 1000) + ' ms for the comparison</span></div>';
}

function render() {
  const ranked = rankedMethods();
  if (!inspected) inspected = ranked[0];
  if (!compared || compared === inspected) compared = ranked.find(method => method !== inspected);
  renderHero(ranked);
  renderFocus();
  renderRanking(ranked);
  renderTradeoffs();
  byId('status').textContent = 'Recommendation: ' + methodShort(ranked[0]) + '. Inspecting ' + methodShort(inspected) + ' against ' + methodShort(compared) + '.';
}

function setTab(tab) {
  const decision = tab === 'decision';
  byId('decision').hidden = !decision;
  byId('methods').hidden = decision;
  byId('decision-tab').setAttribute('aria-selected', String(decision));
  byId('methods-tab').setAttribute('aria-selected', String(!decision));
  byId('decision-tab').tabIndex = decision ? 0 : -1;
  byId('methods-tab').tabIndex = decision ? -1 : 0;
}

byId('priority').onchange = () => {
  inspected = rankedMethods()[0];
  compared = '';
  showAll = false;
  render();
};
byId('compare').onchange = event => { compared = event.target.value; render(); };
byId('target').onchange = renderTradeoffs;
byId('family').onchange = renderTradeoffs;
byId('show-all').onclick = () => { showAll = !showAll; renderRanking(rankedMethods()); };
byId('decision-tab').onclick = () => setTab('decision');
byId('methods-tab').onclick = () => setTab('methods');
byId('stability-link').onclick = event => { event.preventDefault(); setTab('methods'); window.scrollTo(0, 0); };
for (const tab of [byId('decision-tab'), byId('methods-tab')]) {
  tab.onkeydown = event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const other = tab === byId('decision-tab') ? byId('methods-tab') : byId('decision-tab');
      other.click();
      other.focus();
    }
  };
}
setTab(location.hash === '#methods' ? 'methods' : 'decision');
render();
